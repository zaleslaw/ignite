package org.apache.ignite.ml.tree.randomforest;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.ignite.ml.composition.ModelsComposition;
import org.apache.ignite.ml.composition.predictionsaggregator.MeanValuePredictionsAggregator;
import org.apache.ignite.ml.composition.predictionsaggregator.PredictionsAggregator;
import org.apache.ignite.ml.inference.exchange.MLReadable;
import org.apache.ignite.ml.inference.exchange.MLWritable;
import org.apache.ignite.ml.inference.exchange.ModelFormat;
import org.apache.ignite.ml.tree.randomforest.data.RandomForestTreeModel;
import org.dmg.pmml.*;
import org.jpmml.model.PMMLUtil;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class RandomForestModel extends ModelsComposition<RandomForestTreeModel> implements MLReadable, MLWritable {
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
    public RandomForestModel load(Path path, ModelFormat mdlFormat) {
        if (mdlFormat == ModelFormat.PMML) {
            /*try (InputStream is = new FileInputStream(new File(path.toAbsolutePath().toString()))) {
                PMML pmml = PMMLUtil.unmarshal(is);

                TreeModel treeModel = (TreeModel) pmml.getModels().get(0);

                DecisionTreeNode newRootNode = buildTree(treeModel.getNode());
                return new DecisionTreeModel(newRootNode);
            } catch (IOException | JAXBException | SAXException e) {
                e.printStackTrace();
            }*/
        } else if (mdlFormat == ModelFormat.JSON) {
            ObjectMapper mapper = new ObjectMapper();

            RandomForestModel mdl;
            try {
                mdl = mapper.readValue(new File(path.toAbsolutePath().toString()), RandomForestModel.class);

                return mdl;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    @Override
    public void save(Path path, ModelFormat mdlFormat) {
        if (mdlFormat == ModelFormat.PMML) {
        try (OutputStream out = new FileOutputStream(new File(path.toAbsolutePath().toString()))) {
            /*EnsembleModel treeModel = new TreeModel()
                    .setModelName("decision tree")
                    .setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);*/


            Header header = new Header();
            header.setApplication(new Application().setName("Apache Ignite").setVersion("2.9.0-SNAPSHOT"));
            PMML pmml = new PMML(Version.PMML_4_3.getVersion(), header, new DataDictionary())
                    .addModels(null);


            PMMLUtil.marshal(pmml, out);

        } catch (Exception e) {
            e.printStackTrace();
        }

    } else {
        ObjectMapper mapper = new ObjectMapper();

        try {
            File file = new File(path.toAbsolutePath().toString());
            mapper.writeValue(file, this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    }
}
