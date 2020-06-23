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

package org.apache.ignite.ml.tree;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.inference.exchange.MLReadable;
import org.apache.ignite.ml.inference.exchange.MLWritable;
import org.apache.ignite.ml.inference.exchange.ModelFormat;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.dmg.pmml.*;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.*;
import java.nio.file.Path;

/**
 * Base class for decision tree models.
 */
public class DecisionTreeModel implements IgniteModel<Vector, Double>, MLWritable, MLReadable {
    /**
     * Root node.
     */
    private DecisionTreeNode rootNode;

    /**
     * Creates the model.
     *
     * @param rootNode Root node of the tree.
     */
    public DecisionTreeModel(DecisionTreeNode rootNode) {
        this.rootNode = rootNode;
    }

    public DecisionTreeModel() {

    }

    /**
     * Returns the root node.
     */
    public DecisionTreeNode getRootNode() {
        return rootNode;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Double predict(Vector features) {
        return rootNode.predict(features);
    }

    /** {@inheritDoc} */
    @Override public String toString() {
        return toString(false);
    }

    /** {@inheritDoc} */
    @Override public String toString(boolean pretty) {
        return DecisionTreeTrainer.printTree(rootNode, pretty);
    }

    @Override
    public void save(Path path, ModelFormat mdlFormat) {
        if (mdlFormat == ModelFormat.PMML) {
            try (OutputStream out = new FileOutputStream(new File(path.toAbsolutePath().toString()))) {
                TreeModel treeModel = new TreeModel()
                        .setModelName("decision tree")
                        .setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);

                Predicate predicate;

                if (rootNode instanceof DecisionTreeConditionalNode) {
                    DecisionTreeConditionalNode condRootNode = ((DecisionTreeConditionalNode) rootNode);

                    FieldName fieldName = FieldName.create(String.valueOf(condRootNode.getCol()));

                    String threshold = String.valueOf(condRootNode.getThreshold());

                    predicate = new SimplePredicate(fieldName, SimplePredicate.Operator.GREATER_THAN)
                            .setValue(threshold);
                } else {
                    predicate = new True(); // TODO: add test for 1-level tree
                }

                treeModel.setNode(buildPmmlTree(rootNode, predicate));

                Header header = new Header();
                header.setApplication(new Application().setName("Apache Ignite").setVersion("2.9.0-SNAPSHOT"));
                PMML pmml = new PMML(Version.PMML_4_3.getVersion(), header, new DataDictionary())
                        .addModels(treeModel);


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

    @Override
    public DecisionTreeModel load(Path path, ModelFormat mdlFormat) {
        if (mdlFormat == ModelFormat.PMML) {
            try (InputStream is = new FileInputStream(new File(path.toAbsolutePath().toString()))) {
                PMML pmml = PMMLUtil.unmarshal(is);

                TreeModel treeModel = (TreeModel) pmml.getModels().get(0);

                DecisionTreeNode newRootNode = buildTree(treeModel.getNode());
                return new DecisionTreeModel(newRootNode);
            } catch (IOException | JAXBException | SAXException e) {
                e.printStackTrace();
            }
        } else if (mdlFormat == ModelFormat.JSON) {
            ObjectMapper mapper = new ObjectMapper();

            DecisionTreeModel mdl;
            try {
                mdl = mapper.readValue(new File(path.toAbsolutePath().toString()), DecisionTreeModel.class);

                return mdl;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    private DecisionTreeNode buildTree(Node node) {
        Predicate predicate = node.getPredicate();

        if (node.hasNodes()) {
            Node leftNode = null;
            Node rightNode = null;
            for (int i = 0; i < node.getNodes().size(); i++) {
                if(node.getNodes().get(i).getId().equals("left")) {
                    leftNode = node.getNodes().get(i);
                } else if(node.getNodes().get(i).getId().equals("right")) {
                    rightNode = node.getNodes().get(i);
                } else {
                    // TODO: we couldn't handle this case left or right
                }
            }

            int featureIdx = Integer.parseInt(((SimplePredicate)predicate).getField().getValue());
            double threshold = Double.parseDouble(((SimplePredicate)predicate).getValue());

            // TODO: correctly handle missing nodes, add test for that
            String defaultChild = node.getDefaultChild();
            if(defaultChild!= null && !defaultChild.isEmpty()) {
                double missingNodevalue = Double.parseDouble(defaultChild);
                DecisionTreeLeafNode missingNode = new DecisionTreeLeafNode(missingNodevalue);
                return new DecisionTreeConditionalNode(featureIdx, threshold, buildTree(rightNode), buildTree(leftNode), missingNode);
            }
            return new DecisionTreeConditionalNode(featureIdx, threshold, buildTree(rightNode), buildTree(leftNode), null);
        } else {
            return new DecisionTreeLeafNode(Double.parseDouble(node.getScore()));
        }

    }

    private Node buildPmmlTree(DecisionTreeNode node, Predicate predicate) {
        Node pmmlNode = new Node();
        pmmlNode.setPredicate(predicate);

        if (node instanceof DecisionTreeConditionalNode) {
            DecisionTreeConditionalNode splitNode = ((DecisionTreeConditionalNode) node);

            if (splitNode.getMissingNode() != null) {

                DecisionTreeLeafNode missingNode = ((DecisionTreeLeafNode)splitNode.getMissingNode());
                pmmlNode.setDefaultChild(String.valueOf(missingNode.getVal()));
            }

            DecisionTreeNode leftNode = splitNode.getElseNode();
            if(leftNode != null) {
                Predicate leftPredicate = getPredicate(leftNode, true);
                Node leftPmmlNode = buildPmmlTree(leftNode, leftPredicate);
                leftPmmlNode.setId("left");
                pmmlNode.addNodes(leftPmmlNode);
            }

            DecisionTreeNode rightNode = splitNode.getThenNode();
            if(rightNode != null) {
                Predicate rightPredicate = getPredicate(rightNode, false);
                Node rightPmmlNode = buildPmmlTree(rightNode, rightPredicate);
                rightPmmlNode.setId("right");
                pmmlNode.addNodes(rightPmmlNode);
            }
        } else if (node instanceof DecisionTreeLeafNode) {
            DecisionTreeLeafNode leafNode = ((DecisionTreeLeafNode) node);
            pmmlNode.setScore(String.valueOf(leafNode.getVal()));
        }

        return pmmlNode;
    }

    private Predicate getPredicate(DecisionTreeNode node, boolean isLeft) {
        if (node instanceof DecisionTreeConditionalNode) {
            DecisionTreeConditionalNode splitNode = ((DecisionTreeConditionalNode) node);

            FieldName fieldName = FieldName.create(String.valueOf(splitNode.getCol()));

            String threshold = String.valueOf(splitNode.getThreshold());

            if (isLeft) {
                return new SimplePredicate(fieldName, SimplePredicate.Operator.LESS_OR_EQUAL)
                        .setValue(threshold);
            } else {
                return new SimplePredicate(fieldName, SimplePredicate.Operator.GREATER_THAN)
                        .setValue(threshold);
            }
        } else {
            return new True();
        }

    }
}
