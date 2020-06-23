package org.apache.ignite.ml.tree.randomforest;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.ignite.ml.composition.ModelsComposition;
import org.apache.ignite.ml.composition.predictionsaggregator.MeanValuePredictionsAggregator;
import org.apache.ignite.ml.composition.predictionsaggregator.PredictionsAggregator;
import org.apache.ignite.ml.inference.exchange.JSONReadable;
import org.apache.ignite.ml.inference.exchange.JSONWritable;
import org.apache.ignite.ml.tree.randomforest.data.RandomForestTreeModel;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class RandomForestModel extends ModelsComposition<RandomForestTreeModel> implements JSONReadable, JSONWritable {
    /** Serial version uid. */
    private static final long serialVersionUID = 3476345240155508004L;


    public RandomForestModel() {
        super(new ArrayList<>(), new MeanValuePredictionsAggregator());

    }

    public RandomForestModel(List<RandomForestTreeModel> oldModels, PredictionsAggregator predictionsAggregator) {
        super(oldModels, predictionsAggregator);
    }

    /**
     * Returns predictions aggregator.
     */
    @Override
    public PredictionsAggregator getPredictionsAggregator() {
        return predictionsAggregator;
    }

    /**
     * Returns containing models.
     */
    @Override
    public List<RandomForestTreeModel> getModels() {
        return models;
    }

    @Override
    public RandomForestModel fromJSON(Path path) {
            ObjectMapper mapper = new ObjectMapper();

            RandomForestModel mdl;
            try {
                mdl = mapper.readValue(new File(path.toAbsolutePath().toString()), RandomForestModel.class);

                return mdl;
            } catch (IOException e) {
                e.printStackTrace();
            }
        return null;
    }
}
