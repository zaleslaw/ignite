/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.internal.client.thin;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.ignite.client.ClientException;
import org.apache.ignite.client.ClientTransaction;
import org.apache.ignite.client.ClientTransactions;
import org.apache.ignite.configuration.ClientTransactionConfiguration;
import org.apache.ignite.internal.binary.BinaryRawWriterEx;
import org.apache.ignite.internal.binary.BinaryWriterExImpl;
import org.apache.ignite.transactions.TransactionConcurrency;
import org.apache.ignite.transactions.TransactionIsolation;

import static org.apache.ignite.internal.client.thin.ProtocolVersion.V1_5_0;

/**
 * Implementation of {@link ClientTransactions} over TCP protocol.
 */
class TcpClientTransactions implements ClientTransactions {
    /** Transaction label. */
    private String lb;

    /** Channel. */
    private final ReliableChannel ch;

    /** Marshaller. */
    private final ClientBinaryMarshaller marsh;

    /** Current thread transaction id. */
    private final ThreadLocal<Integer> threadLocTxId = new ThreadLocal<>();

    /** Tx map. */
    private final Map<Integer, TcpClientTransaction> txMap = new ConcurrentHashMap<>();

    /** Tx config. */
    private final ClientTransactionConfiguration txCfg;

    /** Constructor. */
    TcpClientTransactions(ReliableChannel ch, ClientBinaryMarshaller marsh, ClientTransactionConfiguration txCfg) {
        this.ch = ch;
        this.marsh = marsh;
        this.txCfg = txCfg;
    }

    /** {@inheritDoc} */
    @Override public ClientTransaction txStart() {
        return txStart0(null, null, null);
    }

    /** {@inheritDoc} */
    @Override public ClientTransaction txStart(TransactionConcurrency concurrency, TransactionIsolation isolation) {
        return txStart0(concurrency, isolation, null);
    }

    /** {@inheritDoc} */
    @Override public ClientTransaction txStart(TransactionConcurrency concurrency, TransactionIsolation isolation,
        long timeout) {
        return txStart0(concurrency, isolation, timeout);
    }

    /**
     * @param concurrency Concurrency.
     * @param isolation Isolation.
     * @param timeout Timeout.
     */
    private ClientTransaction txStart0(TransactionConcurrency concurrency, TransactionIsolation isolation, Long timeout) {
        TcpClientTransaction tx0 = tx();

        if (tx0 != null)
            throw new ClientException("A transaction has already started by the current thread.");

        tx0 = ch.service(ClientOperation.TX_START,
            req -> {
                if (req.clientChannel().serverVersion().compareTo(V1_5_0) < 0) {
                    throw new ClientProtocolError(String.format("Transactions have not supported by the server's " +
                        "protocol version %s, required version %s", req.clientChannel().serverVersion(), V1_5_0));
                }

                try (BinaryRawWriterEx writer = new BinaryWriterExImpl(marsh.context(), req.out(), null, null)) {
                    writer.writeByte((byte)(concurrency == null ? txCfg.getDefaultTxConcurrency() : concurrency).ordinal());
                    writer.writeByte((byte)(isolation == null ? txCfg.getDefaultTxIsolation() : isolation).ordinal());
                    writer.writeLong(timeout == null ? txCfg.getDefaultTxTimeout() : timeout);
                    writer.writeString(lb);
                }
            },
            res -> new TcpClientTransaction(res.in().readInt(), res.clientChannel())
        );

        threadLocTxId.set(tx0.txId);

        txMap.put(tx0.txId, tx0);

        return tx0;
    }

    /** {@inheritDoc} */
    @Override public ClientTransactions withLabel(String lb) {
        if (lb == null)
            throw new NullPointerException();

        TcpClientTransactions txs = new TcpClientTransactions(ch, marsh, txCfg);

        txs.lb = lb;

        return txs;
    }

    /**
     * Current thread transaction.
     */
    TcpClientTransaction tx() {
        Integer txId = threadLocTxId.get();

        if (txId == null)
            return null;

        TcpClientTransaction tx0 = txMap.get(txId);

        // Also check isClosed() flag, since transaction can be closed by another thread.
        return tx0 == null || tx0.isClosed() ? null : tx0;
    }

    /**
     *
     */
    class TcpClientTransaction implements ClientTransaction {
        /** Transaction id. */
        private final int txId;

        /** Client channel. */
        private final ClientChannel clientCh;

        /** Transaction is closed. */
        private volatile boolean closed;

        /**
         * @param id Transaction ID.
         * @param clientCh Client channel.
         */
        private TcpClientTransaction(int id, ClientChannel clientCh) {
            txId = id;
            this.clientCh = clientCh;
        }

        /** {@inheritDoc} */
        @Override public void commit() {
            Integer threadTxId;

            if (closed || (threadTxId = threadLocTxId.get()) == null)
                throw new ClientException("The transaction is already closed");

            if (txId != threadTxId)
                throw new ClientException("You can commit transaction only from the thread it was started");

            endTx(true);
        }

        /** {@inheritDoc} */
        @Override public void rollback() {
            endTx(false);
        }

        /** {@inheritDoc} */
        @Override public void close() {
            try {
                endTx(false);
            }
            catch (Exception ignore) {
                // No-op.
            }
        }

        /**
         * @param committed Committed.
         */
        private void endTx(boolean committed) {
            try {
                ch.service(ClientOperation.TX_END,
                    req -> {
                        if (clientCh != req.clientChannel())
                            throw new ClientException("Transaction context has been lost due to connection errors");

                        req.out().writeInt(txId);
                        req.out().writeBoolean(committed);
                    }, null);
            }
            finally {
                txMap.remove(txId);

                closed = true;

                Integer threadTxId = threadLocTxId.get();

                if (threadTxId != null && txId == threadTxId)
                    threadLocTxId.set(null);
            }
        }

        /**
         * Tx ID.
         */
        int txId() {
            return txId;
        }

        /**
         * Client channel.
         */
        ClientChannel clientChannel() {
            return clientCh;
        }

        /**
         * Is transaction closed.
         */
        boolean isClosed() {
            return closed;
        }
    }
}
