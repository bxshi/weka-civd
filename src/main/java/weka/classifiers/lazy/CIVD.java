/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    IBk.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 <!-- globalinfo-start -->
 * CIVD Classifier<br/>
 * <br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 *
 <!-- options-end -->
 *
 * @author Dash(Baoxu) Shi (bshi@nd.edu)
 * @version $Revision: 10000 $
 */
public class CIVD
        extends AbstractClassifier
        implements OptionHandler, UpdateableClassifier, WeightedInstancesHandler,
        TechnicalInformationHandler, AdditionalMeasureProducer {

    /** for serialization. */
    static final long serialVersionUID = -3080186098777900207L;

    /** The training instances used for classification. */
    protected Instances m_Train;

    /** The number of class values (or 1 if predicting numeric). */
    protected int m_NumClasses;

    /** The class attribute type. */
    protected int m_ClassType;

    /**
     * The maximum number of training instances allowed. When
     * this limit is reached, old training instances are removed,
     * so the training data is "windowed". Set to 0 for unlimited
     * numbers of instances.
     */
    protected int m_WindowSize;

    /** Whether the neighbours should be distance-weighted. */
    protected int m_DistanceWeighting;

    /** Default ZeroR model to use when there are no training instances */
    protected ZeroR m_defaultModel;

    /** for nearest-neighbor search. */
    protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

    /** The number of attributes the contribute to a prediction. */
    protected double m_NumAttributesUsed;

    /**
     * CIVD classifier. All-Nearest Neighbor with gravity-like weight
     *
     */
    public CIVD() {

        init();
    }

    /**
     * Returns a string describing classifier.
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {

        return  "CIVD classifier. \n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;

        result = new TechnicalInformation(Type.ARTICLE);

        return result;
    }

    /**
     * Returns the tip text for this property.
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String windowSizeTipText() {
        return "Gets the maximum number of instances allowed in the training " +
                "pool. The addition of new instances above this value will result " +
                "in old instances being removed. A value of 0 signifies no limit " +
                "to the number of training instances.";
    }

    /**
     * Gets the maximum number of instances allowed in the training
     * pool. The addition of new instances above this value will result
     * in old instances being removed. A value of 0 signifies no limit
     * to the number of training instances.
     *
     * @return Value of WindowSize.
     */
    public int getWindowSize() {

        return m_WindowSize;
    }

    /**
     * Sets the maximum number of instances allowed in the training
     * pool. The addition of new instances above this value will result
     * in old instances being removed. A value of 0 signifies no limit
     * to the number of training instances.
     *
     * @param newWindowSize Value to assign to WindowSize.
     */
    public void setWindowSize(int newWindowSize) {

        m_WindowSize = newWindowSize;
    }

    /**
     * Returns the tip text for this property.
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String nearestNeighbourSearchAlgorithmTipText() {
        return "The nearest neighbour search algorithm to use " +
                "(Default: weka.core.neighboursearch.LinearNNSearch).";
    }

    /**
     * Returns the current nearestNeighbourSearch algorithm in use.
     * @return the NearestNeighbourSearch algorithm currently in use.
     */
    public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
        return m_NNSearch;
    }

    /**
     * Sets the nearestNeighbourSearch algorithm to be used for finding nearest
     * neighbour(s).
     * @param nearestNeighbourSearchAlgorithm - The NearestNeighbourSearch class.
     */
    public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
        m_NNSearch = nearestNeighbourSearchAlgorithm;
    }

    /**
     * Get the number of training instances the classifier is currently using.
     *
     * @return the number of training instances the classifier is currently using
     */
    public int getNumTraining() {

        return m_Train.numInstances();
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
//        result.enable(Capability.NUMERIC_CLASS);
//        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data
     * @throws Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        m_NumClasses = instances.numClasses();
        m_ClassType = instances.classAttribute().type();
        m_Train = new Instances(instances, 0, instances.numInstances());

        // Throw away initial instances until within the specified window size
        if ((m_WindowSize > 0) && (instances.numInstances() > m_WindowSize)) {
            m_Train = new Instances(m_Train,
                    m_Train.numInstances()-m_WindowSize,
                    m_WindowSize);
        }

        System.out.println("buildClassifier, m_Train size "+m_Train.size());

        m_NumAttributesUsed = 0.0;
        for (int i = 0; i < m_Train.numAttributes(); i++) {
            if ((i != m_Train.classIndex()) &&
                    (m_Train.attribute(i).isNominal() ||
                            m_Train.attribute(i).isNumeric())) {
                m_NumAttributesUsed += 1.0;
            }
        }

        m_NNSearch.setInstances(m_Train);

        m_defaultModel = new ZeroR();
        m_defaultModel.buildClassifier(instances);
    }

    /**
     * Adds the supplied instance to the training set.
     *
     * @param instance the instance to add
     * @throws Exception if instance could not be incorporated
     * successfully
     */
    public void updateClassifier(Instance instance) throws Exception {

        if (m_Train.equalHeaders(instance.dataset()) == false) {
            throw new Exception("Incompatible instance types\n" + m_Train.equalHeadersMsg(instance.dataset()));
        }
        if (instance.classIsMissing()) {
            return;
        }

        m_Train.add(instance);
        m_NNSearch.update(instance);
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if an error occurred during the prediction
     */
    public double [] distributionForInstance(Instance instance) throws Exception {

        if (m_Train.numInstances() == 0) {
            //throw new Exception("No training instances!");
            return m_defaultModel.distributionForInstance(instance);
        }

        m_NNSearch.addInstanceInfo(instance);
//        System.out.println("Traning Set size "+m_Train.size());
        Instances neighbours = m_NNSearch.kNearestNeighbours(instance, m_Train.numInstances());
//        System.out.println("Instance Set size "+neighbours.size());
        double [] distances = m_NNSearch.getDistances();
//        System.out.print(instance+" ");
//        for(int i = 0; i < distances.length; i++) {
//            System.out.print(distances[i]+",");
//        }
//        System.out.println("");
        double [] distribution = makeDistribution( neighbours, distances );
        System.out.println("distribution is ");
        for(int i = 0; i < distribution.length; i++) {
            System.out.print(distribution[i]+",");
        }
        System.out.println("");
        return distribution;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>();

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    /**
     * Parses a given list of options. <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     *
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String nnSearchClass = Utils.getOption('A', options);
        if(nnSearchClass.length() != 0) {
            String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
            if(nnSearchClassSpec.length == 0) {
                throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                        "specification string.");
            }
            String className = nnSearchClassSpec[0];
            nnSearchClassSpec[0] = "";

            setNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
                            Utils.forName( NearestNeighbourSearch.class,
                                    className,
                                    nnSearchClassSpec)
            );
        }
        else
            this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());

        super.setOptions(options);

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of IBk.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String [] getOptions() {

        Vector<String> options = new Vector<String>();

        options.add("-A");
        options.add(m_NNSearch.getClass().getName()+" "+Utils.joinOptions(m_NNSearch.getOptions()));

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    /**
     * Returns an enumeration of the additional measure names
     * produced by the neighbour search algorithm, plus the chosen K in case
     * cross-validation is enabled.
     *
     * @return an enumeration of the measure names
     */
    public Enumeration<String> enumerateMeasures() {
            return m_NNSearch.enumerateMeasures();
    }

    /**
     * Returns the value of the named measure from the
     * neighbour search algorithm, plus the chosen K in case
     * cross-validation is enabled.
     *
     * @param additionalMeasureName the name of the measure to query for its value
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {
            return m_NNSearch.getMeasure(additionalMeasureName);
    }


    /**
     * Returns a description of this classifier.
     *
     * @return a description of this classifier as a string.
     */
    public String toString() {

        if (m_Train == null) {
            return "CIVD: No model built yet.";
        }

        if (m_Train.numInstances() == 0) {
            return "Warning: no training instances - ZeroR model used.";
        }

        String result = "CIVD classifier\n";

        return result;
    }

    /**
     * Initialise scheme variables.
     */
    protected void init() {

        m_WindowSize = 0;
    }

    /**
     * Turn the list of nearest neighbors into a probability distribution.
     *
     * @param neighbours the list of nearest neighboring instances
     * @param distances the distances of the neighbors
     * @return the probability distribution
     * @throws Exception if computation goes wrong or has no class attribute
     */
    protected double [] makeDistribution(Instances neighbours, double[] distances)
            throws Exception {

        double total = 0, weight;
        double [] distribution = new double [m_NumClasses];
        int [] classMember = new int [m_NumClasses];

        for(int i = 0; i < m_NumClasses; i++) {
            classMember[i] = 0;
        }

        // Set up a correction to the estimator
        if (m_ClassType == Attribute.NOMINAL) {
            for(int i = 0; i < m_NumClasses; i++) {
                distribution[i] = 1.0 / Math.max(1,m_Train.numInstances());
            }
        }

        for(int i=0; i < neighbours.numInstances(); i++) {
            // Collect class counts
            Instance current = neighbours.instance(i);
            distances[i] = distances[i]*distances[i];
            weight = 1.0 / (distances[i] + Double.MIN_VALUE); // to avoid div by zero
            weight *= current.weight();

            try {
                switch (m_ClassType) {
                    case Attribute.NOMINAL:
                        distribution[(int)current.classValue()] += weight;
                        classMember[(int)current.classValue()]++;
                        break;
//                    case Attribute.NUMERIC:
//                        distribution[0] += current.classValue() * weight;
//                        break;
                }
            } catch (Exception ex) {
                throw new Error("Data has no class attribute!");
            }
            total += weight;
        }

        double max = 0.0;
        int maxIdx = -1;
        for(int i = 0; i < distribution.length; i++) {
            distribution[i] = distribution[i] / classMember[i];
            if(distribution[i] > max) {
                max = distribution[i];
                if(maxIdx != -1) {
                    distribution[maxIdx] = 0;
                }
                maxIdx = i;
                distribution[i] = 1;
            }else{
                distribution[i] = 0;
            }
        }
        // Normalise distribution
        return distribution;
    }

    /**
     * Prunes the list to contain the k nearest neighbors. If there are
     * multiple neighbors at the k'th distance, all will be kept.
     *
     * @param neighbours the neighbour instances.
     * @param distances the distances of the neighbours from target instance.
     * @param k the number of neighbors to keep.
     * @return the pruned neighbours.
     */
    public Instances pruneToK(Instances neighbours, double[] distances, int k) {

        if(neighbours==null || distances==null || neighbours.numInstances()==0) {
            return null;
        }
        if (k < 1) {
            k = 1;
        }

        int currentK = 0;
        double currentDist;
        for(int i=0; i < neighbours.numInstances(); i++) {
            currentK++;
            currentDist = distances[i];
            if(currentK>k && currentDist!=distances[i-1]) {
                currentK--;
                neighbours = new Instances(neighbours, 0, currentK);
                break;
            }
        }

        return neighbours;
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10141 $");
    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain command line options (see setOptions)
     */
    public static void main(String [] argv) {
        runClassifier(new IBk(), argv);
    }
}
