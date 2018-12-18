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
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.dataset.Dataset;
import org.apache.ignite.ml.dataset.DatasetBuilder;
import org.apache.ignite.ml.dataset.PartitionDataBuilder;
import org.apache.ignite.ml.dataset.primitive.context.EmptyContext;
import org.apache.ignite.ml.environment.LearningEnvironmentBuilder;
import org.apache.ignite.ml.math.Tracer;
import org.apache.ignite.ml.math.distances.DistanceMeasure;
import org.apache.ignite.ml.math.distances.EuclideanDistance;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.structures.LabeledVector;
import org.apache.ignite.ml.structures.LabeledVectorSet;
import org.apache.ignite.ml.structures.partition.LabeledDatasetPartitionDataBuilderOnHeap;
import org.apache.ignite.ml.trainers.SingleLabelDatasetTrainer;

/**
 * The trainer for DBSCAN algorithm.
 *
 * NOTE: test more than 1000 rows
 */
public class DBSCANTrainer extends SingleLabelDatasetTrainer<DBSCANModel> {
    public static final int MAX_AMOUNT_OF_CLUSTERS = 1_000_000_000;
    /** Minimum number of points needed for a cluster. */
    private int minimumNumOfClusterMembers = 10;

    /** Maximum radius of the neighborhood to be considered. */
    private double epsilon = 1f;

    /** Distance measure. */
    private DistanceMeasure distanceMeasure = new EuclideanDistance();

    /** KMeans initializer. */
    private long seed;

    /** Status of a point during the clustering process. */
    private enum VectorStatus {
        /** The point has is considered to be noise. */
        NOISE,
        /** The point is already part of a cluster. */
        PART_OF_CLUSTER
    }

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
    @Override public DBSCANTrainer withEnvironmentBuilder(LearningEnvironmentBuilder envBuilder) {
        return (DBSCANTrainer)super.withEnvironmentBuilder(envBuilder);
    }

    /** {@inheritDoc} */
    @Override protected <K, V> DBSCANModel updateModel(DBSCANModel mdl, DatasetBuilder<K, V> datasetBuilder,
        IgniteBiFunction<K, V, Vector> featureExtractor, IgniteBiFunction<K, V, Double> lbExtractor) {

        assert datasetBuilder != null;

        PartitionDataBuilder<K, V, EmptyContext, LabeledVectorSet<Double, LabeledVector>> partDataBuilder = new LabeledDatasetPartitionDataBuilderOnHeap<>(
            featureExtractor,
            lbExtractor
        );

        try (Dataset<EmptyContext, LabeledVectorSet<Double, LabeledVector>> dataset = datasetBuilder.build(
            envBuilder,
            (env, upstream, upstreamSize) -> new EmptyContext(),
            partDataBuilder
        )) {
  /*          final Integer cols = dataset.compute(org.apache.ignite.ml.structures.Dataset::colSize, (a, b) -> {
                if (a == null)
                    return b == null ? 0 : b;
                if (b == null)
                    return a;
                return b;
            });*/

           /* if (cols == null)
                return getLastTrainedModelOrThrowEmptyDatasetException(mdl);*/

            dataset.compute((data, env) -> {
                Map<LabeledVector, VectorStatus> visitedPoints = new HashMap<>();
                double clsLb = env.partition() * MAX_AMOUNT_OF_CLUSTERS;

                Map<Double, List<LabeledVector>> clusters = new HashMap<>(); // class label and list of class members

                for (int i = 0; i < data.rowSize(); i++) {
                    LabeledVector pnt = data.getRow(i);

                    if (!visitedPoints.containsKey(pnt)) {
                        List<LabeledVector> neighbours = getNeighbors(pnt, data);

                        if (neighbours.size() > minimumNumOfClusterMembers) {
                            clsLb += 1;
                            clusters.put(clsLb, new ArrayList<>());
                            expandCluster(clusters.get(clsLb), pnt, neighbours, data, visitedPoints);
                        }
                        else
                            visitedPoints.put(pnt, VectorStatus.NOISE);
                    }
                }

                clusters.forEach(
                    (k, v) -> {

                        try {
                            Thread.sleep(1000);
                        }
                        catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                        System.out.println("Cluster: " + k + " size " + v.size());
                        v.forEach(e -> {
                            System.out.println("Class" + e.label());
                            // System.out.print("Guid" + e.features().guid());

                            Tracer.showAscii(e.features());
                        });
                    }
                );

                //reduce amount of points in each cluster (drop граница или внутренности?) или просто проредить точки сгущения?
                // это может быть параметром approximate coeeficient при коээфициенте равном 1 - все точки перемещаем, чтобы склеить кластера
                // а может быть можно переместить упакованную информацию о кластерах и ее склеить - это граница в евклидовом смысле
                // можно рандомно, монте-карлово брать точки из попарных кластеров, чтобы проверить гипотезу о склейке или жестко упаковать инфу о точках в какой-то binary format
                // нужна процедура внутри каждой партиции - поиск опорных точек в каждом кластере, просто рандомная подвыборка для перемешения дальше не больше k экземпляров, например
                // причем желательно не нарушать основное свойство кластера из вики

                // т.е. входной параметр - не более k кандидатов как в ANN число кластеров, если k = 0 или <0 то берем всех кандататов и тащим через reduce
                // есть опасность, что они все упададут в какие-то точки згущения - можно после рандомной генерации проверять, что не соседи *10 попыток - правда так и кластеры могут распаться, если старым способом клеить
                // а если по новому клеить - типо по близости, то все норм м.б.
                return clusters;
            }, null);

        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
        return new DBSCANModel(distanceMeasure);
    }

    private List<LabeledVector> expandCluster(List<LabeledVector> cluster,
        LabeledVector point,
        List<LabeledVector> neighbors,
        LabeledVectorSet<Double, LabeledVector> points,
        Map<LabeledVector, VectorStatus> visitedPoints) {

        cluster.add(point);
        visitedPoints.put(point, VectorStatus.PART_OF_CLUSTER);

        List<LabeledVector> neighborsCp = new ArrayList<>(neighbors);

        for (int i = 0; i < neighbors.size(); i++) {
            LabeledVector curr = neighbors.get(i);
            VectorStatus status = visitedPoints.get(curr);

            if (status == null) {
                List<LabeledVector> currentNeighbors = getNeighbors(curr, points);
                if (currentNeighbors.size() >= minimumNumOfClusterMembers)
                    neighborsCp = merge(neighborsCp, currentNeighbors);
            }
            if (status != VectorStatus.PART_OF_CLUSTER) {
                visitedPoints.put(curr, VectorStatus.PART_OF_CLUSTER);
                cluster.add(curr);
            }
        }
        return cluster;
    }

    private List<LabeledVector> merge(List<LabeledVector> one, List<LabeledVector> two) {
        final Set<LabeledVector> oneSet = new HashSet<>(one);
        for (LabeledVector item : two) {
            if (!oneSet.contains(item))
                one.add(item);
        }
        return one;
    }

    private List<LabeledVector> getNeighbors(LabeledVector point, LabeledVectorSet<Double, LabeledVector> vectorSet) {
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
    public DBSCANTrainer withEpsilon(double epsilon) {
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
    public DBSCANTrainer withDistance(DistanceMeasure distance) {
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
    public DBSCANTrainer withSeed(long seed) {
        this.seed = seed;
        return this;
    }
}
