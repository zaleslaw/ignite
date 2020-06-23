package org.apache.ignite.ml.composition.boosting;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.composition.ModelsComposition;
import org.apache.ignite.ml.composition.predictionsaggregator.WeightedPredictionsAggregator;
import org.apache.ignite.ml.inference.exchange.MLReadable;
import org.apache.ignite.ml.inference.exchange.MLWritable;
import org.apache.ignite.ml.inference.exchange.ModelFormat;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.tree.DecisionTreeModel;
import org.dmg.pmml.*;
import org.jpmml.model.PMMLUtil;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.List;

/**
 * GDB model.
 */
public final class GDBModel extends ModelsComposition<DecisionTreeModel> implements MLReadable, MLWritable {
    /** Serial version uid. */
    private static final long serialVersionUID = 3476661240155508004L;

    /** Internal to external lbl mapping. */
    @JsonIgnore private IgniteFunction<Double, Double> internalToExternalLblMapping;

    /**
     * Creates an instance of GDBModel.
     *
     * @param models Models.
     * @param predictionsAggregator Predictions aggregator.
     * @param internalToExternalLblMapping Internal to external lbl mapping.
     */
    public GDBModel(List<? extends IgniteModel<Vector, Double>> models,
                    WeightedPredictionsAggregator predictionsAggregator,
                    IgniteFunction<Double, Double> internalToExternalLblMapping) {

        super((List<DecisionTreeModel>) models, predictionsAggregator);
        this.internalToExternalLblMapping = internalToExternalLblMapping;
    }

    public GDBModel() {
    }

    public GDBModel withLblMapping(IgniteFunction<Double, Double> internalToExternalLblMapping) {
        this.internalToExternalLblMapping = internalToExternalLblMapping;
        return this;
    }

    /** {@inheritDoc} */
    @Override public Double predict(Vector features) {
        if(internalToExternalLblMapping == null) {
            throw new IllegalArgumentException("The mapping should not be empty. Initialize it with apropriate function. ");
        } else {
            return internalToExternalLblMapping.apply(super.predict(features));
        }
    }

    @Override public GDBModel load(Path path, ModelFormat mdlFormat) {
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

            GDBModel mdl;
            try {
                mdl = mapper.readValue(new File(path.toAbsolutePath().toString()), GDBModel.class);

                return mdl;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    @Override public void save(Path path, ModelFormat mdlFormat) {
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
