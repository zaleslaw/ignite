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

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.UUID;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.impl.DenseVector;

/**
 * The utility class.
 */
public class DatasetGenerator {
    public static final int ROW_SIZE = 14;
    public static final int AMOUNT_OF_ROWS = 10_000_000;
    private static Random rnd1 = new Random(1L);
    private static Random rnd2 = new Random(2L);
    private static Random rnd3 = new Random(3L);
    private static Random rnd4 = new Random(4L);
    private static Random rnd5 = new Random(5L);
    private static Random rnd6 = new Random(6L);
    private static Random rnd7 = new Random(7L);
    private static Random rnd8 = new Random(8L);
    private static Random rnd9 = new Random(9L);
    private static Random rnd10 = new Random(10L);
    private static Random rnd11 = new Random(11L);
    private static Random rnd12 = new Random(12L);
    private static Random rnd13 = new Random(13L);
    private static Random rnd14 = new Random(14L);

    public static IgniteCache<Integer, Vector> loadDataset(Ignite ignite) {
        IgniteCache<Integer, Vector> cache = getCache(ignite);
        double avgRegVal = 0.0;

        for (int i = 0; i < AMOUNT_OF_ROWS; i++) {

            Serializable[] data = initializeData(i);
            cache.put(i++, new DenseVector(data));
            avgRegVal += (double)data[13];
        }

        System.out.println("average reg val is " + (avgRegVal/AMOUNT_OF_ROWS));
        return cache;
    }

    /**
     * Fills cache with data and returns it.
     *
     * @param ignite Ignite instance.
     * @return Filled Ignite Cache.
     */
    private static IgniteCache<Integer, Vector> getCache(Ignite ignite) {

        CacheConfiguration<Integer, Vector> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName("Multiplan_" + UUID.randomUUID());
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        return ignite.createCache(cacheConfiguration);
    }

    private static Serializable[] initializeData(int i) {
        Serializable[] data = new Serializable[ROW_SIZE];
        data[0] =  (double)rnd1.nextInt(24) + 1; // ML_TAT
        data[1] = rnd2.nextDouble() * 2030; //CLIENTMAXTARGETAMOUNT
        data[2] = rnd3.nextDouble(); //ML_SAVINGRATE_PROVIDER
        data[3] = rnd4.nextDouble(); //ML_SUCCESSRATE_PROVIDER
        data[4] = rnd5.nextBoolean() ? 1.0 : 0.0; // ML_REASON_CLIENTLIABILITYLOWERTHANTHRESHHOLD
        data[5] = rnd6.nextDouble(); // ML_SUCCESSRATE_CHARGES_PROVIDER
        data[6] = rnd7.nextDouble() * 495 * 2; // IMP_DIS_DIP
        data[7] =  (double)rnd8.nextInt(74) + 1; // ML_CPTCOUNT
        data[8] = rnd9.nextDouble() * 0.91; // ML_SAVINGRATE_CLIENT
        data[9] = rnd10.nextDouble(); //ML_SUCCESSRATE_CHARGES_CLIENT
        data[10] = (double)rnd11.nextInt(100);// ML_CLAIM_DURATION
        data[11]  = "category" + rnd12.nextInt(4);// PROVIDERTYPEID
        data[12] = rnd13.nextDouble();// ML_SUCCESSRATE_CLIENT
        data[13]  = getRegressionValueBasedOnFeatures(data); //ML_SAVING_NEW

        //System.out.println(Arrays.toString(data));

        return data;
    }

    // 429 000 as max value
    private static Serializable getRegressionValueBasedOnFeatures(Serializable[] data) {
        double regVal = rnd14.nextDouble() * 1000 * 2;
        if ((double)data[0] > 20.0) regVal += 1000;
        if ((double)data[1] < 10) regVal += 1000;
        if ((double)data[2] > 0.9) regVal += 1000;
        if ((double)data[3] > 0.95) regVal += 10000;
        if ((double)data[4] == 0.0) regVal+=500;
        if ((double)data[5] < 0.001) regVal+= 10000;
        if ((double)data[6] > 300) regVal+= 500;
        if ((double)data[7] > 73) regVal+= 1000;
        if ((double)data[8] < 0.3) regVal+= 100;
        if ((double)data[9] > 0.1) regVal += 100;
        if ((double)data[10]> 99) regVal+= 10000;
        if ((String.valueOf(data[11]).equals("category1"))) regVal+= 1000;
        if ((String.valueOf(data[11]).equals("category2"))) regVal+= 100;
        if ((String.valueOf(data[11]).equals("category3"))) regVal+= 10;
        if ((double)data[12] > 0.8) regVal+= 100;

        return regVal;
    }
}
