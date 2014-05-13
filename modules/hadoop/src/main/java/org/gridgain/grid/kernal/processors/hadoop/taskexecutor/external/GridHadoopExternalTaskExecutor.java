/* @java.file.header */

/*  _________        _____ __________________        _____
 *  __  ____/___________(_)______  /__  ____/______ ____(_)_______
 *  _  / __  __  ___/__  / _  __  / _  / __  _  __ `/__  / __  __ \
 *  / /_/ /  _  /    _  /  / /_/ /  / /_/ /  / /_/ / _  /  _  / / /
 *  \____/   /_/     /_/   \_,__/   \____/   \__,_/  /_/   /_/ /_/
 */

package org.gridgain.grid.kernal.processors.hadoop.taskexecutor.external;

import org.gridgain.grid.*;
import org.gridgain.grid.hadoop.*;
import org.gridgain.grid.kernal.*;
import org.gridgain.grid.kernal.processors.hadoop.*;
import org.gridgain.grid.kernal.processors.hadoop.jobtracker.*;
import org.gridgain.grid.kernal.processors.hadoop.message.*;
import org.gridgain.grid.kernal.processors.hadoop.taskexecutor.*;
import org.gridgain.grid.kernal.processors.hadoop.taskexecutor.external.child.*;
import org.gridgain.grid.kernal.processors.hadoop.taskexecutor.external.communication.*;
import org.gridgain.grid.logger.*;
import org.gridgain.grid.spi.*;
import org.gridgain.grid.util.*;
import org.gridgain.grid.util.future.*;
import org.gridgain.grid.util.typedef.*;
import org.gridgain.grid.util.typedef.internal.*;
import org.jdk8.backport.*;
import org.jetbrains.annotations.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;

import static org.gridgain.grid.kernal.processors.hadoop.taskexecutor.GridHadoopTaskState.*;

/**
 * External process registry. Handles external process lifecycle.
 */
public class GridHadoopExternalTaskExecutor extends GridHadoopTaskExecutorAdapter {
    /** Hadoop context. */
    private GridHadoopContext ctx;

    /** */
    private String javaCmd;

    /** Logger. */
    private GridLogger log;

    /** Node process descriptor. */
    private GridHadoopProcessDescriptor nodeDesc;

    /** Output base. */
    private String outputBase;

    /** Path separator. */
    private String pathSep;

    /** Hadoop external communication. */
    private GridHadoopExternalCommunication comm;

    /** Starting processes. */
    private ConcurrentMap<UUID, GridHadoopProcessFuture> startingProcs =
        new ConcurrentHashMap8<>();

    /** Starting processes. */
    private ConcurrentMap<UUID, HadoopProcess> runningProcsByProcId =
        new ConcurrentHashMap8<>();

    /** Starting processes. */
    private ConcurrentMap<GridHadoopJobId, HadoopProcess> runningProcsByJobId =
        new ConcurrentHashMap8<>();

    /** Busy lock. */
    private GridSpinReadWriteLock busyLock = new GridSpinReadWriteLock();

    /** Job tracker. */
    private GridHadoopJobTracker jobTracker;

    /** {@inheritDoc} */
    @Override public void start(GridHadoopContext ctx) throws GridException {
        this.ctx = ctx;

        log = ctx.kernalContext().log(GridHadoopExternalTaskExecutor.class);

        outputBase = U.getGridGainHome() + File.separator + "work" + File.separator + "hadoop";

        pathSep = System.getProperty("path.separator", U.isWindows() ? ";" : ":");

        initJavaCommand();

        comm = new GridHadoopExternalCommunication(
            ctx.localNodeId(),
            UUID.randomUUID(),
            ctx.kernalContext().config().getMarshaller(),
            log,
            ctx.kernalContext().config().getSystemExecutorService(),
            ctx.kernalContext().gridName());

        comm.setSharedMemoryPort(-1); // TODO.

        comm.setListener(new MessageListener());

        comm.start();

        nodeDesc = comm.localProcessDescriptor();

        ctx.kernalContext().ports().registerPort(nodeDesc.tcpPort(), GridPortProtocol.TCP,
            GridHadoopExternalTaskExecutor.class);

        if (nodeDesc.sharedMemoryPort() != -1)
            ctx.kernalContext().ports().registerPort(nodeDesc.sharedMemoryPort(), GridPortProtocol.TCP,
                GridHadoopExternalTaskExecutor.class);

        jobTracker = ctx.jobTracker();
    }

    /** {@inheritDoc} */
    @Override public void stop(boolean cancel) {
        busyLock.writeLock();

        try {
            comm.stop();
        }
        catch (GridException e) {
            U.error(log, "Failed to gracefully stop external hadoop communication server (will shutdown anyway)", e);
        }
    }

    /** {@inheritDoc} */
    @Override public void run(final GridHadoopJob job, final Collection<GridHadoopTaskInfo> tasks) {
        if (!busyLock.tryReadLock()) {
            if (log.isDebugEnabled())
                log.debug("Failed to start hadoop tasks (grid is stopping, will ignore).");

            return;
        }

        try {
            GridHadoopExternalTaskMetadata taskMeta = buildTaskMeta(job);

            if (log.isDebugEnabled())
                log.debug("Created hadoop task metadata for job [job=" + job + ", taskMeta=" + taskMeta + ']');

            startProcess(taskMeta).listenAsync(new CI1<GridFuture<HadoopProcess>>() {
                @Override public void apply(GridFuture<HadoopProcess> procFut) {
                    if (!busyLock.tryReadLock())
                        return;

                    try {
                        HadoopProcess proc = procFut.get();

                        if (log.isDebugEnabled())
                            log.debug("Child process was initialized, sending execution request [jobId=" + job.id() +
                                ", tasksSize=" + tasks.size() + ']');

                        proc.jobId(job.id());
                        proc.tasks(tasks);

                        runningProcsByJobId.put(job.id(), proc);

                        sendExecutionRequest(proc, job, tasks);
                    }
                    catch (GridException e) {
                        U.error(log, "Failed to start external task process for Hadoop job: " + job, e);

                        notifyTasksFailed(tasks, FAILED, e);
                    }
                    finally {
                        busyLock.readUnlock();
                    }
                }
            });
        }
        finally {
            busyLock.readUnlock();
        }
    }

    /** {@inheritDoc} */
    @Override public void cancelTasks(GridHadoopJobId jobId) {
        // TODO: implement.
    }

    /** {@inheritDoc} */
    @Override public void onJobStateChanged(GridHadoopJobId jobId, GridHadoopJobMetadata meta) {
        HadoopProcess proc = runningProcsByJobId.get(jobId);

        // If we have a local task process for job running.
        if (proc != null) {
            if (log.isDebugEnabled())
                log.debug("Updating job information for remote task process [proc=" + proc + ", meta=" + meta + ']');

            try {
                Map<Integer, GridHadoopProcessDescriptor> rdcAddrs = meta.reducersAddresses();

                int rdcNum = meta.mapReducePlan().reducers();

                GridHadoopProcessDescriptor[] addrs = null;

                if (rdcAddrs != null && rdcAddrs.size() == rdcNum) {
                    addrs = new GridHadoopProcessDescriptor[rdcNum];

                    for (int i = 0; i < rdcNum; i++) {
                        GridHadoopProcessDescriptor desc = rdcAddrs.get(i);

                        assert desc != null : "Missing reducing address [meta=" + meta + ", rdc=" + i + ']';

                        addrs[i] = desc;
                    }
                }

                comm.sendMessage(proc.descriptor(), new GridHadoopJobInfoUpdateRequest(jobId, meta.phase(), addrs));
            }
            catch (GridException e) {
                // TODO fail the whole thing?
            }
        }
    }

    /**
     * Sends execution request to remote node.
     *
     * @param proc Process to send request to.
     * @param job Job instance.
     * @param tasks Collection of tasks to execute in started process.
     */
    private void sendExecutionRequest(HadoopProcess proc, GridHadoopJob job, Collection<GridHadoopTaskInfo> tasks)
        throws GridException {
        // Must synchronize since concurrent process crash may happen and will receive onConnectionLost().
        proc.lock();

        try {
            if (proc.terminated()) {
                notifyTasksFailed(tasks, CRASHED, null);

                return;
            }

            GridHadoopTaskExecutionRequest req = new GridHadoopTaskExecutionRequest();

            req.jobId(job.id());
            req.jobInfo(job.info());
            req.tasks(tasks);
            req.concurrentMappers(8);  // TODO
            req.concurrentReducers(8); // TODO

            comm.sendMessage(proc.descriptor(), req);
        }
        finally {
            proc.unlock();
        }
    }

    /**
     * @param job Job instance.
     * @return External task metadata.
     */
    private GridHadoopExternalTaskMetadata buildTaskMeta(GridHadoopJob job) {
        GridHadoopExternalTaskMetadata meta = new GridHadoopExternalTaskMetadata();

        // TODO how to build classpath? Deploy task jar here.
        meta.classpath(Arrays.asList(System.getProperty("java.class.path").split(":")));
        meta.jvmOptions(Arrays.asList("-Xmx1g", "-ea" ,"-DGRIDGAIN_HOME=" + U.getGridGainHome()));

        return meta;
    }

    /**
     * @param tasks Tasks to notify about.
     * @param state Fail state.
     * @param e Optional error.
     */
    private void notifyTasksFailed(Iterable<GridHadoopTaskInfo> tasks, GridHadoopTaskState state, Throwable e) {
        GridHadoopTaskStatus fail = new GridHadoopTaskStatus(state, e);

        for (GridHadoopTaskInfo task : tasks)
            jobTracker.onTaskFinished(task, fail);
    }

    /**
     * Starts process template that will be ready to execute Hadoop tasks.
     *
     * @param meta Process metadata.
     * @return Start future.
     */
    private GridFuture<HadoopProcess> startProcess(final GridHadoopExternalTaskMetadata meta) {
        final UUID childProcId = UUID.randomUUID();

        final GridHadoopProcessFuture fut = new GridHadoopProcessFuture(childProcId, ctx.kernalContext());

        GridHadoopProcessFuture old = startingProcs.put(childProcId, fut);

        assert old == null;

        ctx.kernalContext().closure().runLocalSafe(new Runnable() {
            @Override public void run() {
                if (!busyLock.tryReadLock()) {
                    fut.onDone(new GridException("Failed to start external process (grid is stopping)."));

                    return;
                }

                try {
                    if (log.isDebugEnabled())
                        log.debug("Starting process based on task metadata: " + meta);

                    Process proc = initializeProcess(childProcId, meta);

                    BufferedReader rdr = new BufferedReader(new InputStreamReader(proc.getInputStream()));

                    String line;

                    // Read up all the process output.
                    while ((line = rdr.readLine()) != null) {
                        if (log.isDebugEnabled())
                            log.debug("Tracing process output: " + line);

                        if ("Started".equals(line)) {
                            // Process started successfully, it should not write anything more to the output stream.
                            if (log.isDebugEnabled())
                                log.debug("Successfully started child process [childProcId=" + childProcId +
                                    ", meta=" + meta + ']');

                            fut.onProcessStarted(proc);

                            break;
                        }
                        else if ("Failed".equals(line)) {
                            StringBuilder sb = new StringBuilder("Failed to start child process: " + meta + "\n");

                            while ((line = rdr.readLine()) != null)
                                sb.append("    ").append(line).append("\n");

                            // Cut last character.
                            sb.setLength(sb.length() - 1);

                            log.warning(sb.toString());

                            fut.onDone(new GridException(sb.toString()));

                            break;
                        }
                    }
                }
                catch (Throwable e) {
                    fut.onDone(new GridException("Failed to initialize child process: " + meta, e));
                }
                finally {
                    busyLock.readUnlock();
                }
            }
        }, true);

        return fut;
    }

    /**
     * Checks that java local command is available.
     *
     * @throws GridException If initialization failed.
     */
    private void initJavaCommand() throws GridException {
        String javaHome = System.getProperty("java.home");

        if (javaHome == null)
            javaHome = System.getenv("JAVA_HOME");

        if (javaHome == null)
            throw new GridException("Failed to locate JAVA_HOME.");

        javaCmd = javaHome + File.separator + "bin" + File.separator + (U.isWindows() ? "java.exe" : "java");

        try {
            Process proc = new ProcessBuilder(javaCmd, "-version").redirectErrorStream(true).start();

            Collection<String> out = readProcessOutput(proc);

            int res = proc.waitFor();

            if (res != 0)
                throw new GridException("Failed to execute 'java -version' command (process finished with nonzero " +
                    "code) [exitCode=" + res + ", javaCmd='" + javaCmd + "', msg=" + F.first(out) + ']');

            if (log.isInfoEnabled()) {
                log.info("Will use java for external task execution: ");

                for (String s : out)
                    log.info("    " + s);
            }
        }
        catch (IOException e) {
            throw new GridException("Failed to check java for external task execution.", e);
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();

            throw new GridException("Failed to wait for process completion (thread got interrupted).", e);
        }
    }

    /**
     * Reads process output line-by-line.
     *
     * @param proc Process to read output.
     * @return Read lines.
     * @throws IOException If read failed.
     */
    private Collection<String> readProcessOutput(Process proc) throws IOException {
        BufferedReader rdr = new BufferedReader(new InputStreamReader(proc.getInputStream()));

        Collection<String> res = new ArrayList<>();

        String s;

        while ((s = rdr.readLine()) != null)
            res.add(s);

        return res;
    }

    /**
     * Builds process from metadata.
     *
     * @param childProcId Child process ID.
     * @param meta Metadata.
     * @return Started process.
     */
    private Process initializeProcess(UUID childProcId, GridHadoopExternalTaskMetadata meta) throws Exception {
        List<String> cmd = new ArrayList<>();

        cmd.add(javaCmd);
        cmd.addAll(meta.jvmOptions());
        cmd.add("-cp");
        cmd.add(buildClasspath(meta.classpath()));
        cmd.add(GridHadoopExternalProcessStarter.class.getName());
        cmd.add("-cpid");
        cmd.add(String.valueOf(childProcId));
        cmd.add("-ppid");
        cmd.add(String.valueOf(nodeDesc.processId()));
        cmd.add("-nid");
        cmd.add(String.valueOf(nodeDesc.parentNodeId()));
        cmd.add("-addr");
        cmd.add(nodeDesc.address());
        cmd.add("-tport");
        cmd.add(String.valueOf(nodeDesc.tcpPort()));
        cmd.add("-sport");
        cmd.add(String.valueOf(nodeDesc.sharedMemoryPort()));
        cmd.add("-out");
        cmd.add(outputBase + File.separator + childProcId);

        return new ProcessBuilder(cmd)
            .redirectErrorStream(true)
            .start();
    }

    /**
     * @param cp Classpath collection.
     * @return Classpath string.
     */
    private String buildClasspath(Collection<String> cp) {
        assert !cp.isEmpty();

        StringBuilder sb = new StringBuilder();

        for (String s : cp)
            sb.append(s).append(pathSep);

        sb.setLength(sb.length() - 1);

        return sb.toString();
    }

    /**
     * Processes task finished message.
     *
     * @param desc Remote process descriptor.
     * @param taskMsg Task finished message.
     */
    private void processTaskFinishedMessage(GridHadoopProcessDescriptor desc, GridHadoopTaskFinishedMessage taskMsg) {
        HadoopProcess proc = runningProcsByProcId.get(desc.processId());

        if (proc != null)
            proc.removeTask(taskMsg.taskInfo());

        jobTracker.onTaskFinished(taskMsg.taskInfo(), new GridHadoopTaskStatus(taskMsg.state(), taskMsg.error()));
    }

    /**
     *
     */
    private class MessageListener implements GridHadoopMessageListener {
        /** {@inheritDoc} */
        @Override public void onMessageReceived(GridHadoopProcessDescriptor desc, GridHadoopMessage msg) {
            if (!busyLock.tryReadLock())
                return;

            try {
                if (msg instanceof GridHadoopProcessStartedAck) {
                    GridHadoopProcessFuture fut = startingProcs.get(desc.processId());

                    if (fut != null)
                        fut.onReplyReceived(desc);
                    // Safety.
                    else
                        log.warning("Failed to find process start future (will ignore): " + desc);
                }
                else if (msg instanceof GridHadoopTaskFinishedMessage) {
                    GridHadoopTaskFinishedMessage taskMsg = (GridHadoopTaskFinishedMessage)msg;

                    processTaskFinishedMessage(desc, taskMsg);
                }
                else if (msg instanceof GridHadoopTaskExecutionResponse) {
                    GridHadoopTaskExecutionResponse res = (GridHadoopTaskExecutionResponse)msg;

                    if (!F.isEmpty(res.reducers()))
                        jobTracker.onExternalMappersInitialized(res.jobId(), res.reducers(), desc);
                }
                else
                    log.warning("Unexpected message received by node [desc=" + desc + ", msg=" + msg + ']');
            }
            finally {
                busyLock.readUnlock();
            }
        }

        /** {@inheritDoc} */
        @Override public void onConnectionLost(GridHadoopProcessDescriptor desc) {
            if (!busyLock.tryReadLock())
                return;

            try {
                // Cleanup. Gracefully terminated process should be cleaned up by this moment.
                HadoopProcess proc = runningProcsByProcId.remove(desc.processId());

                if (proc != null) {
                    log.warning("Lost connection with alive process (will terminate): " + desc);

                    Collection<GridHadoopTaskInfo> tasks = proc.tasks();

                    if (!F.isEmpty(tasks)) {
                        GridHadoopTaskStatus status = new GridHadoopTaskStatus(CRASHED,
                            new GridException("Failed to run tasks (external process finished unexpectedly): " + desc));

                        for (GridHadoopTaskInfo info : tasks)
                            jobTracker.onTaskFinished(info, status);
                    }

                    runningProcsByJobId.remove(proc.jobId());

                    proc.terminate();
                }
            }
            finally {
                busyLock.readUnlock();
            }
        }
    }

    /**
     * Hadoop process.
     */
    private static class HadoopProcess extends ReentrantLock {
        /** Job ID. */
        private GridHadoopJobId jobId;

        /** Process. */
        private Process proc;

        /** Process descriptor. */
        private GridHadoopProcessDescriptor procDesc;

        /** Tasks. */
        private Collection<GridHadoopTaskInfo> tasks;

        /** Terminated flag. */
        private boolean terminated;

        /**
         * @param proc Java process representation.
         * @param procDesc Process communication descriptor.
         */
        private HadoopProcess(Process proc, GridHadoopProcessDescriptor procDesc) {
            this.proc = proc;
            this.procDesc = procDesc;
        }

        /**
         * @return Java process representation.
         */
        private Process process() {
            return proc;
        }

        /**
         * @return Communication process descriptor.
         */
        private GridHadoopProcessDescriptor descriptor() {
            return procDesc;
        }

        /**
         * @return Job ID.
         */
        public GridHadoopJobId jobId() {
            return jobId;
        }

        /**
         * @param jobId Job ID.
         */
        public void jobId(GridHadoopJobId jobId) {
            this.jobId = jobId;
        }

        /**
         * Terminates process (kills it).
         */
        private void terminate() {
            lock();

            try {
                terminated = true;

                proc.destroy();
            }
            finally {
                unlock();
            }
        }

        /**
         * @return Terminated flag.
         */
        private boolean terminated() {
            return terminated;
        }

        /**
         * Sets process tasks.
         *
         * @param tasks Tasks to set.
         */
        private void tasks(Collection<GridHadoopTaskInfo> tasks) {
            this.tasks = new ConcurrentLinkedDeque<>(tasks);
        }

        /**
         * Removes task when it was completed.
         *
         * @param task Task to remove.
         */
        private void removeTask(GridHadoopTaskInfo task) {
            if (tasks != null)
                tasks.remove(task);
        }

        /**
         * @return Collection of tasks.
         */
        private Collection<GridHadoopTaskInfo> tasks() {
            return tasks;
        }

        /** {@inheritDoc} */
        @Override public String toString() {
            return S.toString(HadoopProcess.class, this);
        }
    }

    /**
     *
     */
    private class GridHadoopProcessFuture extends GridFutureAdapter<HadoopProcess> {
        /** Child process ID. */
        private UUID childProcId;

        /** Process descriptor. */
        private GridHadoopProcessDescriptor desc;

        /** Running process. */
        private Process proc;

        /** Process started flag. */
        private volatile boolean procStarted;

        /** Reply received flag. */
        private volatile boolean replyReceived;

        /**
         * Empty constructor.
         */
        public GridHadoopProcessFuture() {
            // No-op.
        }

        /**
         * @param ctx Kernal context.
         */
        private GridHadoopProcessFuture(UUID childProcId, GridKernalContext ctx) {
            super(ctx);

            this.childProcId = childProcId;
        }

        /**
         * Process started callback.
         */
        public void onProcessStarted(Process proc) {
            this.proc = proc;

            procStarted = true;

            if (procStarted && replyReceived)
                onDone(new HadoopProcess(proc, desc));
        }

        /**
         * Reply received callback.
         */
        public void onReplyReceived(GridHadoopProcessDescriptor desc) {
            assert childProcId.equals(desc.processId());

            this.desc = desc;

            replyReceived = true;

            if (procStarted && replyReceived)
                onDone(new HadoopProcess(proc, desc));
        }

        /** {@inheritDoc} */
        @Override public boolean onDone(@Nullable HadoopProcess res, @Nullable Throwable err) {
            if (super.onDone(res, err)) {
                startingProcs.remove(childProcId, this);

                HadoopProcess old = runningProcsByProcId.put(childProcId, res);

                runningProcsByJobId.put(res.jobId, res);

                assert old == null;

                return true;
            }

            return false;
        }
    }
}
