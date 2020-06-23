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

package org.apache.ignite.examples.ml.inference.exportimport;

import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.examples.ml.util.MLSandboxDatasets;
import org.apache.ignite.examples.ml.util.SandboxMLCache;
import org.apache.ignite.ml.clustering.kmeans.KMeansModel;
import org.apache.ignite.ml.clustering.kmeans.KMeansTrainer;
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.math.Tracer;
import org.apache.ignite.ml.math.primitives.vector.Vector;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Run KMeans clustering algorithm ({@link KMeansTrainer}) over distributed dataset.
 * <p>
 * Code in this example launches Ignite grid and fills the cache with test data points (based on the
 * <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set"></a>Iris dataset</a>).</p>
 * <p>
 * After that it trains the model based on the specified data using
 * <a href="https://en.wikipedia.org/wiki/K-means_clustering">KMeans</a> algorithm.</p>
 * <p>
 * Finally, this example loops over the test set of data points, applies the trained model to predict what cluster does
 * this point belong to, and compares prediction to expected outcome (ground truth).</p>
 * <p>
 * You can change the test data used in this example and re-run it to explore this algorithm further.</p>
 */
public class KMeansClusterizationExample2 {
    /**
     * Run example.
     */
    public static void main(String[] args) throws IOException {
        System.out.println();
        System.out.println(">>> KMeans clustering algorithm over cached dataset usage example started.");
        // Start ignite grid.
        try (Ignite ignite = Ignition.start("examples/config/example-ignite.xml")) {
            System.out.println(">>> Ignite grid started.");

            IgniteCache<Integer, Vector> dataCache = null;
            try {
                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.TWO_CLASSED_IRIS);

                Vectorizer<Integer, Vector, Integer, Double> vectorizer = new DummyVectorizer<Integer>().labeled(Vectorizer.LabelCoordinate.FIRST);

                KMeansTrainer trainer = new KMeansTrainer();

                KMeansModel mdl = trainer.fit(
                    ignite,
                    dataCache,
                    vectorizer
                );

                System.out.println(">>> KMeans centroids");
                Tracer.showAscii(mdl.centers()[0]);
                Tracer.showAscii(mdl.centers()[1]);
                System.out.println(">>>");

                Path pmmlMdlPath = Paths.get("C:\\ignite\\kmeans.pmml");
                mdl.toPMML(pmmlMdlPath); // TODO: write to the root in tmp directory

                Path jsonMdlPath = Paths.get("C:\\ignite\\kmeans.json");
                mdl.toJSON(jsonMdlPath); // TODO: write to the root in tmp directory

                System.out.println(">>> kmeans model " + mdl);

                KMeansModel pmmlMdl = new KMeansModel().fromPMML(pmmlMdlPath);
                System.out.println(">>> pmml kmeans model " + pmmlMdl);

                KMeansModel jsonMdl = new KMeansModel().fromJSON(jsonMdlPath);
                System.out.println(">>> json kmeans model " + jsonMdl);

            }
            finally {
                if (dataCache != null)
                    dataCache.destroy();
            }
        }
        finally {
            System.out.flush();
        }
    }
}
