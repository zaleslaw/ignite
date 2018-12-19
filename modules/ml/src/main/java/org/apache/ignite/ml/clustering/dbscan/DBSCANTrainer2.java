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

package org.apache.ignite.ml.clustering.dbscan;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.ignite.IgniteAtomicLong;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.clustering.VectorStatus;
import org.apache.ignite.ml.dataset.Dataset;
import org.apache.ignite.ml.dataset.DatasetBuilder;
import org.apache.ignite.ml.dataset.PartitionDataBuilder;
import org.apache.ignite.ml.dataset.impl.cache.CacheBasedDatasetBuilder;
import org.apache.ignite.ml.dataset.primitive.context.EmptyContext;
import org.apache.ignite.ml.environment.LearningEnvironmentBuilder;
import org.apache.ignite.ml.math.distances.DistanceMeasure;
import org.apache.ignite.ml.math.distances.EuclideanDistance;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteBinaryOperator;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.structures.LabeledVector;
import org.apache.ignite.ml.structures.partition.LabeledDatasetPartitionDataBuilderOnHeap2;
import org.apache.ignite.ml.structures.partition.LabeledVectorSet2;
import org.apache.ignite.ml.trainers.SingleLabelDatasetTrainer;
import org.jetbrains.annotations.NotNull;

/**
 * The trainer for DBSCAN algorithm.
 *
 * NOTE: test more than 1000 rows
 *
 * NOTE: This implementation uses random partitioning
 *
 * NOTE: the minimumNumOfClusterMembers in each partition should be calculated as minimumNumOfClusterMembers/amount of partitions
 * and epsilon should be recalculated according amount of partitions
 */
public class DBSCANTrainer2 extends SingleLabelDatasetTrainer<DBSCANModel> {
    /** Minimum number of points needed for a cluster. */
    private int minimumNumOfClusterMembers = 100;

    /** Maximum radius of the neighborhood to be considered. */
    private double epsilon = 1.0;

    /** Distance measure. */
    private DistanceMeasure distanceMeasure = new EuclideanDistance();

    /** KMeans initializer. */
    private long seed;

    /**
     * Trains model based on the specified data.
     *
     * @param datasetBuilder Dataset builder.
     * @param featureExtractor Feature extractor.
     * @param lbExtractor Label extractor.
     * @return Model.
     */
    @Override public <K, V> DBSCANModel fit(DatasetBuilder<K, V> datasetBuilder,
        IgniteBiFunction<K, V, Vector> featureExtractor, IgniteBiFunction<K, V, Double> lbExtractor) {

        return updateModel(null, datasetBuilder, featureExtractor, lbExtractor);
    }

    /** {@inheritDoc} */
    @Override public DBSCANTrainer2 withEnvironmentBuilder(LearningEnvironmentBuilder envBuilder) {
        return (DBSCANTrainer2)super.withEnvironmentBuilder(envBuilder);
    }

    private List<DBSCANVector> getGlobalNeighbors(
        Dataset<EmptyContext, LabeledVectorSet2<Double, DBSCANVector>> dataset,
        DBSCANVector pnt, IgniteBinaryOperator<List<DBSCANVector>> mergeLocalNeighbors) {
        return dataset.compute(
            (vectorSet, env2) -> findLocalNeighbors(pnt, vectorSet),
            mergeLocalNeighbors
        );
    }

    @NotNull
    private List<DBSCANVector> findLocalNeighbors(DBSCANVector pnt, LabeledVectorSet2<Double, DBSCANVector> vectorSet) {
        List<DBSCANVector> neighbors = new ArrayList<>();

        for (int j = 0; j < vectorSet.rowSize(); j++) {
            DBSCANVector neighborCandidate = vectorSet.getRow(j);
            if (pnt != neighborCandidate && distanceMeasure.compute(pnt.features(), neighborCandidate.features()) <= epsilon)
                neighbors.add(neighborCandidate);
        }
        return neighbors;
    }

    /** {@inheritDoc} */
    @Override protected <K, V> DBSCANModel updateModel(DBSCANModel mdl, DatasetBuilder<K, V> datasetBuilder,
        IgniteBiFunction<K, V, Vector> featureExtractor, IgniteBiFunction<K, V, Double> lbExtractor) {

        assert datasetBuilder != null;

        PartitionDataBuilder<K, V, EmptyContext, LabeledVectorSet2<Double, DBSCANVector>> partDataBuilder = new LabeledDatasetPartitionDataBuilderOnHeap2<>(
            featureExtractor,
            lbExtractor
        );

        try (Dataset<EmptyContext, LabeledVectorSet2<Double, DBSCANVector>> dataset = datasetBuilder.build(
            envBuilder,
            (env, upstream, upstreamSize) -> new EmptyContext(),
            partDataBuilder
        )) {

            IgniteAtomicLong clusterCntr = ((CacheBasedDatasetBuilder)datasetBuilder).getIgnite().atomicLong("cluster counter", 0, true);

            dataset.compute((data, env) -> {
                int clsLb;
                for (int i = 0; i < data.rowSize(); i++) {
                    DBSCANVector pnt = (DBSCANVector)data.getRow(i);
                    if (pnt.status == VectorStatus.UNKNOWN) {

                        IgniteBinaryOperator<List<DBSCANVector>> mergeLocNeighbors = (a, b) -> {
                            if (a == null)
                                return b == null ? new ArrayList<>() : b;
                            if (b == null)
                                return a;
                            a.addAll(b); // due to unqie keys in local DBSCAN clusterization
                            return a;
                        };

                        List<DBSCANVector> globalNeighbors = getGlobalNeighbors(dataset, pnt, mergeLocNeighbors);
                        // how long it will be correct? All the time!!! Due to neighboring is stable state from start to end of fit

                        if (globalNeighbors.size() > minimumNumOfClusterMembers) {
                            clsLb = (int)clusterCntr.incrementAndGet();
                            expandCluster(clsLb, pnt, globalNeighbors, dataset, mergeLocNeighbors);
                        }
                        else
                            pnt.status = VectorStatus.NOISE;
                    }
                }
                return null;
            }, null);

            dataset.compute((data, env) -> {
                System.out.println("Partition " + env.partition());
                Map<Integer, Integer> cntrs1 = new HashMap<>();
                Map<Integer, Integer> cntrs2 = new HashMap<>();

                for (int i = 0; i < data.rowSize(); i++) {
                    DBSCANVector vector = data.getRow(i);
                    int clusterId = vector.clusterId;

                    if ((double)vector.label() == 1.0) {

                        if (cntrs1.containsKey(clusterId)) {
                            int cnt = cntrs1.get(clusterId);
                            cnt++;
                            cntrs1.put(clusterId, cnt);
                        }
                        else {
                            cntrs1.put(clusterId, 1);
                        }
                    }
                    if ((double)vector.label() == 2.0) {
                        if (cntrs2.containsKey(clusterId)) {
                            int cnt = cntrs2.get(clusterId);
                            cnt++;
                            cntrs2.put(clusterId, cnt);
                        }
                        else {
                            cntrs2.put(clusterId, 1);
                        }
                    }

                    //System.out.println(vector.clusterId + " " + vector.status + " " + vector.label());

                }

                System.out.println("Class 1.0 Distribution");
                cntrs1.forEach((k, v) -> System.out.println(k + " " + v));
                System.out.println("Class 2.0 Distribution");
                cntrs2.forEach((k, v) -> System.out.println(k + " " + v));
                return null;
            }, null);
            clusterCntr.close();

        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
        return new DBSCANModel(distanceMeasure);
    }

    private void expandCluster(int lb, DBSCANVector pnt, List<DBSCANVector> neighbors,
        Dataset<EmptyContext, LabeledVectorSet2<Double, DBSCANVector>> dataset,
        IgniteBinaryOperator<List<DBSCANVector>> mergeLocalNeighbors) {
        pnt.clusterId = lb;
        pnt.status = VectorStatus.PART_OF_CLUSTER;

        List<DBSCANVector> neighborsCp = new ArrayList<>(neighbors);

        int idx = 0;
        while (idx < neighborsCp.size()) {
            DBSCANVector curr = neighbors.get(idx);
            VectorStatus status = curr.status;

            if (status == VectorStatus.UNKNOWN) {
                List<DBSCANVector> currNeighbors = getGlobalNeighbors(dataset, curr, mergeLocalNeighbors);
                if (currNeighbors.size() >= minimumNumOfClusterMembers)
                    ;
                neighborsCp = merge(neighborsCp, currNeighbors); // TODO: incorrect merge
            }
            if (status != VectorStatus.PART_OF_CLUSTER) {
                curr.status = VectorStatus.PART_OF_CLUSTER;
                curr.clusterId = lb;
            }
            idx++;
        }
    }

    private List<DBSCANVector> merge(List<DBSCANVector> one, List<DBSCANVector> two) {
        final Set<DBSCANVector> oneSet = new HashSet<>(one);
        for (DBSCANVector item : two) {
            if (!oneSet.contains(item))
                one.add(item);
        }
        return one;
    }

    // TODO: cache all calculated distances as a parameter but it requires O(n2) memory
    private List<LabeledVector> getNeighbors(LabeledVector point, LabeledVectorSet2<Double, DBSCANVector> vectorSet) {
        List<LabeledVector> neighbors = new ArrayList<>();

        for (int i = 0; i < vectorSet.rowSize(); i++) {
            LabeledVector neighborCandidate = vectorSet.getRow(i);
            if (point != neighborCandidate && distanceMeasure.compute(point.features(), neighborCandidate.features()) <= epsilon)
                neighbors.add(neighborCandidate);
        }

        return neighbors;
    }

    /** {@inheritDoc} */
    @Override protected boolean checkState(DBSCANModel mdl) {
        return mdl.distanceMeasure().equals(distanceMeasure);
    }

    /**
     * Find the closest cluster center index and distanceMeasure to it from a given point.
     *
     * @param centers Centers to look in.
     * @param pnt Point.
     */
    private IgniteBiTuple<Integer, Double> findClosestCentroid(Vector[] centers, LabeledVector pnt) {
        double bestDistance = Double.POSITIVE_INFINITY;
        int bestInd = 0;

        for (int i = 0; i < centers.length; i++) {
            double dist = distanceMeasure.compute(centers[i], pnt.features());
            if (dist < bestDistance) {
                bestDistance = dist;
                bestInd = i;
            }
        }
        return new IgniteBiTuple<>(bestInd, bestDistance);
    }

    /**
     * Gets the epsilon.
     *
     * @return The parameter value.
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Set up the epsilon.
     *
     * @param epsilon The parameter value.
     * @return Model with new epsilon parameter value.
     */
    public DBSCANTrainer2 withEpsilon(double epsilon) {
        this.epsilon = epsilon;
        return this;
    }

    /**
     * Gets the distanceMeasure.
     *
     * @return The parameter value.
     */
    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    /**
     * Set up the distanceMeasure.
     *
     * @param distance The parameter value.
     * @return Model with new distanceMeasure parameter value.
     */
    public DBSCANTrainer2 withDistance(DistanceMeasure distance) {
        this.distanceMeasure = distance;
        return this;
    }

    /**
     * Gets the seed number.
     *
     * @return The parameter value.
     */
    public long getSeed() {
        return seed;
    }

    /**
     * Set up the seed.
     *
     * @param seed The parameter value.
     * @return Model with new seed parameter value.
     */
    public DBSCANTrainer2 withSeed(long seed) {
        this.seed = seed;
        return this;
    }
}
