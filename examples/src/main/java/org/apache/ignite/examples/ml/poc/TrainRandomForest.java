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

package org.apache.ignite.examples.ml.poc;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.examples.ml.tutorial.Step_1_Read_and_Learn;
import org.apache.ignite.ml.composition.ModelsComposition;
import org.apache.ignite.ml.dataset.feature.FeatureMeta;
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.environment.LearningEnvironmentBuilder;
import org.apache.ignite.ml.environment.logging.ConsoleLogger;
import org.apache.ignite.ml.environment.parallelism.ParallelismStrategy;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.preprocessing.Preprocessor;
import org.apache.ignite.ml.preprocessing.encoding.EncoderTrainer;
import org.apache.ignite.ml.preprocessing.encoding.EncoderType;
import org.apache.ignite.ml.selection.scoring.evaluator.Evaluator;
import org.apache.ignite.ml.selection.scoring.metric.MetricName;
import org.apache.ignite.ml.selection.split.TrainTestDatasetSplitter;
import org.apache.ignite.ml.selection.split.TrainTestSplit;
import org.apache.ignite.ml.tree.randomforest.RandomForestRegressionTrainer;
import org.apache.ignite.ml.tree.randomforest.data.FeaturesCountSelectionStrategies;

/**
 * Let's add two categorial features "sex", "embarked" to predict more precisely than in {@link
 * Step_1_Read_and_Learn}..
 * <p>
 * To encode categorial features the {@link EncoderTrainer} of the
 * <a href="https://en.wikipedia.org/wiki/One-hot">One-hot</a> type will be used.</p>
 * <p>
 * Code in this example launches Ignite grid and fills the cache with test data (based on Titanic passengers data).</p>
 * <p>
 * After that it defines preprocessors that extract features from an upstream data and encode string values (categories)
 * to double values in specified range.</p>
 * <p>
 * Then, it trains the model based on the processed data using decision tree classification.</p>
 * <p>
 * Finally, this example uses {@link Evaluator} functionality to compute metrics from predictions.</p>
 */
public class TrainRandomForest {
    /**
     * Run example.
     */
    public static void main(String[] args) {
        try (Ignite ignite = Ignition.start("examples/config/example-ignite.xml")) {

            IgniteCache<Integer, Vector> dataCache = DatasetGenerator.loadDataset(ignite);

            TrainTestSplit<Integer, Vector> split = new TrainTestDatasetSplitter<Integer, Vector>()
                .split(0.9);

            final Vectorizer<Integer, Vector, Integer, Double> vectorizer = new DummyVectorizer<Integer>()
                .labeled(Vectorizer.LabelCoordinate.LAST);

            Preprocessor<Integer, Vector> oneHotEncoderPreprocessor = new EncoderTrainer<Integer, Vector>()
                .withEncoderType(EncoderType.ONE_HOT_ENCODER)
                .withEncodedFeature(11)
                .fit(ignite,
                    dataCache,
                    vectorizer
                );

            AtomicInteger idx = new AtomicInteger(0);
            int amountOfFeatures = DatasetGenerator.ROW_SIZE + 3; // basic features + amount of one-hot-encoded columns
            List<FeatureMeta> meta = IntStream.range(0, amountOfFeatures - 1)
                .mapToObj(x -> new FeatureMeta("", idx.getAndIncrement(), false)).collect(Collectors.toList());

            RandomForestRegressionTrainer trainer = new RandomForestRegressionTrainer(meta)
                .withAmountOfTrees(100)
                .withFeaturesCountSelectionStrgy(FeaturesCountSelectionStrategies.ALL)
                .withMaxDepth(5)
                .withMinImpurityDelta(0.)
                .withSubSampleSize(0.33)
                .withSeed(0);

            trainer.withEnvironmentBuilder(LearningEnvironmentBuilder.defaultBuilder()
                .withParallelismStrategyTypeDependency(ParallelismStrategy.NO_PARALLELISM)
                .withLoggingFactoryDependency(ConsoleLogger.Factory.LOW)
            );

            long startTraining = System.currentTimeMillis();

            ModelsComposition randomForestMdl = trainer.fit(
                ignite,
                dataCache,
                split.getTrainFilter(),
                oneHotEncoderPreprocessor
            );


            long endTraining = System.currentTimeMillis();

            System.out.println("\n>>> Trained model: " + randomForestMdl.toString());
            System.out.println("Training time is " + (endTraining - startTraining));

            double mae = Evaluator.evaluate(
                dataCache,
                split.getTestFilter(),
                randomForestMdl,
                oneHotEncoderPreprocessor,
                MetricName.MAE
            );

            System.out.println("\n>>> MAE " + mae);

            long endEvaluation = System.currentTimeMillis();
            System.out.println("Evaluation time is " + (endEvaluation - endTraining));

        }
        finally {
            System.out.flush();
        }
    }
}
