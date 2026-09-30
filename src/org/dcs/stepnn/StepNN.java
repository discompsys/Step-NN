/*
 * StepNN.java
 *
 * Created on 14 October 2025
 *
 * Version 1.0
 *
 * Copyright (c) Kieran Greer
 */

package org.dcs.stepnn;


import org.ai_heuristic.data.Dataset;
import org.ai_heuristic.eval.metric.ReplySet;
import org.ai_heuristic.eval.result.Cluster;
import org.ai_heuristic.eval.result.ClusterResult;
import org.ai_heuristic.util.AiHeuristicConst;
import org.jlog2.util.UuidHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;


/**
 * New version that adds new points as steps to outliers.
 */
public abstract class StepNN {

    /** Indicates no entry index */
    public static int NOENTRY = -111111111;


    /** Evaluation metric type */
    protected String evalType;

    /** The dataset rows to cluster in the algorithm */
    protected ArrayList<double[]> dataRows;

    /** List of data row labels for each row */
    protected ArrayList<String> categories;

    /** The dataset for training that models the best result */
    protected ArrayList<double[]> trainRows;

    /**
     * List of clusters with conflicting categories.
     */
    protected ClusterResult notMatch;

    /**
     * List of ordered distance keys.
     */
    protected ArrayList<Double> distOrdered;

    /**
     * Distances to a point.
     */
    protected HashMap<Double, ArrayList<Integer>> myDistances;


    /**
     * Create a new instance of StepNN_v1.
     * @param evalType the type of evaluation metric,
     *                 for example {@codeAiHeuristicConst.EUCLIDEANFUNCTION},
     *                 see {@code evaluate}.
     * @param ds the dataset.
     * @throws Exception any error.
     */
    public StepNN(String evalType, Dataset ds) throws Exception {

        int i;
        HashMap<String, ArrayList<Integer>> dsKeys;     //dataset keys

        this.evalType = evalType;
        dataRows = ds.getDataset();
        categories = new ArrayList<>();
        notMatch = null;

        for (i = 0; i < ds.getDataSize(); i++) {
            categories.add(null);
        }

        dsKeys = ds.getCategories();
        for (String cKey : dsKeys.keySet()) {
            for (Integer index : dsKeys.get(cKey)) {
                categories.set(index, cKey);
            }
        }

        trainRows = (ArrayList<double[]>) dataRows.clone();
    }

    /**
     * Evaluate the closest data rows to the selected one, using a distance metric.
     * @param rowNumber The row number for the selected row.
     * @param md the row values.
     * @param testSet if new test dataset, then include all rows, even same index.
     * @return list of k nearest rows.
     * @throws Exception any error.
     */
    protected HashMap<Double, ArrayList<Integer>> predict(int rowNumber, double[] md, boolean testSet) throws Exception {

        int i;
        double distance;
        HashMap<Double, ArrayList<Integer>> closest;        //distances
        double[] row1, row2;                                //data rows

        closest = new HashMap<>();

        for (i = 0; i < trainRows.size(); i++) {
            if (testSet || (i != rowNumber)) {
                row1 = trainRows.get(i);
                row2 = md;

                distance = evaluate(row1, row2, evalType);
                if (!closest.containsKey(distance)) {
                    closest.put(distance, new ArrayList<>());
                }

                closest.get(distance).add(i);
            }
        }

        return closest;
    }

    /**
     * Calculate the closest ksize data points from each point to the other, but use the train dataset for
     * measuring this.
     * @param dataset the dataset to cluster, as specified above.
     * @return the cluster indexes from the {@code datasets} list, as {@code ArrayList} lists.
     * You can use {@code ((Integer)ReplySet.getValue()).intValue()} to retrieve the cluster values.
     * @throws Exception any error.
     */
    public ReplySet testOnly(ArrayList<double[]> dataset,
                             ArrayList<double[]> trainRows,
                             ArrayList<String> trnCategories) throws Exception {
        int i;
        String myLabel;                                         //categories
        ArrayList<Integer> indexList;                           //index list
        ArrayList<Integer> posList;                             //index list
        ArrayList<Integer> cluster;                             //a cluster
        HashMap<Double, ArrayList<Integer>> myDistances;        //distances to a point
        HashMap<Integer, ArrayList<Integer>> clusters;          //list of clusters
        double[] mv;                                            //metric value

        clusters = new HashMap<>();
        this.trainRows = trainRows;

        //generate all distances from one point to every other point
        for (i = 0; i < dataset.size(); i++)
        {
            mv = dataset.get(i);
            myDistances = predict(i, mv, false);

            cluster = new ArrayList<>();
            cluster.add(i);
            myLabel = categories.get(i);

            indexList = closestDist(myDistances);
            posList = matchList(myLabel, indexList, trnCategories);

            if (isMatch(indexList, posList)) {
                cluster.addAll(posList);
            }
            else {
                cluster.add(NOENTRY);
            }

            clusters.put(i, cluster);
        }

        return ReplySet.toSet(toClusters(clusters));
    }

    /**
     * Evaluate the distance using some metric.
     * @param ds1 first row.
     * @param ds2 second row.
     * @param metric metric type.
     * @return distance between row 1 and row 2.
     * @throws Exception any error.
     */
    protected double evaluate(double[] ds1, double[] ds2, String metric) throws Exception {

        switch (metric) {
            case AiHeuristicConst.EUCLIDEANFUNCTION:
                return euclidean(ds1, ds2);
            case AiHeuristicConst.CITYBLOCKFUNCTION:
                return cityBlock(ds1, ds2);
            case AiHeuristicConst.COSINEFUNCTION:
                return cosine(ds1, ds2);
            default:
                return euclidean(ds1, ds2);
        }
    }

    /**
     * Get the index for the closest distance.
     * @param myDistances list of all distances.
     * @return index for row with closest distance.
     */
    protected ArrayList<Integer> closestDist(HashMap<Double, ArrayList<Integer>> myDistances) {
        double bestDist;

        bestDist = Double.MAX_VALUE;
        for (Double nextDist : myDistances.keySet()) {
            if (nextDist < bestDist) {
                bestDist = nextDist;
            }
        }

        return myDistances.get(bestDist);
    }

    /**
     * Determine if a train dataset point with same label is closest, or majority closest.
     * @param indexList list of row indexes for current distance.
     * @param posList list of positive matches from the index list (call matchList).
     * @return true if closest, false if other category closest.
     * @throws Exception any error.
     */
    protected boolean isMatch(ArrayList<Integer> indexList,
                              ArrayList<Integer> posList) {
        int pos;                            //pos count

        pos = (indexList.size() / 2) + 1;
        return (posList.size() >= pos);
    }

    /**
     * Create a list of points with same label.
     * @param myLabel point label to match with.
     * @param indexList list of row indexes for current distance.
     * @param trnCategories full list of categories for each train dataset point.
     * @return list of points with same label.
     * @throws Exception any error.
     */
    protected ArrayList<Integer> matchList(String myLabel,
                                           ArrayList<Integer> indexList,
                                           ArrayList<String> trnCategories) throws Exception {
        String nextLabel;                           //categories
        ArrayList<Integer> cluster;                 //a cluster

        cluster = new ArrayList<>();
        for (int index : indexList) {
            nextLabel = trnCategories.get(index);
            if (myLabel.equalsIgnoreCase(nextLabel)) {
                cluster.add(index);
            }
        }

        return cluster;
    }

    /**
     * Determine what clusters do not match 100% with the same category.
     * @param nnClusters the knn clusters result.
     *                   First is from the test list, rest from train list.
     * @return clusters that contain more than 1 category.
     * @throws Exception any error.
     */
    public ClusterResult notMatch(ArrayList<ArrayList<Integer>> nnClusters,
                                  ArrayList<String> trnCategories,
                                  ArrayList<String> tstCategories)
            throws Exception {
        int i;
        int trueCount, falseCount;          //counts
        String cat1, cat2;                  //categories
        ArrayList<Integer> nnCluster;       //cluster
        Cluster cluster;                    //cluster

        notMatch = new ClusterResult();

        for (i = 0; i < nnClusters.size(); i++) {
            nnCluster = nnClusters.get(i);
            trueCount = 0;
            falseCount = 0;
            cat1 = null;

            for (Integer index : nnCluster) {
                if (cat1 == null) {
                    cat1 = tstCategories.get(index);
                }
                else {
                    if (index == NOENTRY) {
                        falseCount++;
                    }
                    else {
                        cat2 = trnCategories.get(index);
                        if ((cat2 == null) || !cat2.equals(cat1)) {
                            falseCount++;
                        } else {
                            trueCount++;
                        }
                    }
                }
            }

            if (falseCount > trueCount) {
                cluster = new Cluster();
                cluster.name = UuidHandler.getUuid(5);
                cluster.list = nnCluster;
                notMatch.clusters.put(String.valueOf(i), cluster);
            }
        }

        return notMatch;
    }

    /**
     * Evaluate the comparison between the two data lists using a Euclidean distance.
     * If each list has only one object, then apply to the objects directly and not the lists.
     * @param ds1 first value dataset.
     * @param ds2 second value dataset.
     * @return the result of applying the operator to the values.
     */
    protected double euclidean(double[] ds1, double[] ds2)
    {
        int i;
        double distance;                        //distance

        distance = 0.0;
        for (i = 0; i < ds1.length; i++)
        {
            distance += (Math.pow((ds1[i] - ds2[i]), 2.0));
        }

        return Math.sqrt(distance);
    }


    /**
     * Evaluate the comparison between the two data lists and return the result.
     * If each list has only one object, then apply to the objects directly and not the lists.
     * @param ds1 first value dataset.
     * @param ds2 second value dataset.
     * @return the result of applying the operator to the values.
     * @throws Exception any error.
     */
    protected double cityBlock(double[] ds1, double[] ds2) throws Exception
    {
        int i;
        double distance;                        //distance

        distance = 0.0;
        for (i = 0; i < ds1.length; i++)
        {
            distance += (Math.abs(ds1[i] - ds2[i]));
        }

        return distance;
    }

    /**
     * Evaluate the comparison between the two data lists using a cosine distance.
     * If each list has only one object, then apply to the objects directly and not the lists.
     * @param ds1 first value dataset.
     * @param ds2 second value dataset.
     * @return the result of applying the operator to the values.
     * @throws Exception any error.
     */
    protected double cosine(double[] ds1, double[] ds2) throws Exception
    {
        int i;
        double dotproduct;                      //dot product of the two vectors
        double magnitude1;                      //magnitude of the two vectors
        double magnitude2;                      //magnitude of the two vectors
        double magnitude;                       //magnitude of the two vectors
        double result;                          //result of evaluation

        magnitude1 = 0.0;
        magnitude2 = 0.0;
        dotproduct = 0.0;

        for (i = 0; i < ds1.length; i++)
        {
            magnitude1 += Math.abs(ds1[i]);
            magnitude2 += Math.abs(ds2[i]);
            dotproduct += (ds1[i] * ds2[i]);
        }

        magnitude = (magnitude1 * magnitude2);

        //the cosine similarity is then the dotproduct divided by the magnitude
        //if dotproduct is null then return a double value of 0
        //if magnitude is null then do not divide by it
        if (dotproduct == 0) result = 0.0;
        else if (magnitude == 0) result = dotproduct;
        else result = (dotproduct / magnitude);

        return (1.0 - result);
    }

    /**
     * Save network into file
     * @param fileName File Name
     */
    public void dataToFile(String fileName) {
        String rowStr;
        double[] dataRow;
        File outFile = new File(fileName);

        try {
            FileWriter fw = new FileWriter(outFile);
            PrintWriter pw = new PrintWriter(fw);

            for (int i = 0; i < trainRows.size(); i++) {
                dataRow = trainRows.get(i);
                rowStr = "";

                for (int j = 0; j < dataRow.length; j++) {
                    rowStr += (dataRow[j] + ", ");
                }

                rowStr += categories.get(i);
                pw.println(rowStr);
            }

            fw.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Convert the groups of nn nodes to clusters of node names.
     * @param nnGroups the groups of nn nodes to cluster.
     * @return the clustered names lists.
     */
    public ArrayList<ArrayList<Integer>> toClusters(HashMap<Integer, ArrayList<Integer>> nnGroups)
    {
        Integer nextKey;                                //next key
        Iterator<Integer> allKeys;                      //all keys
        ArrayList<Integer> nextGroup;                   //next group
        ArrayList<ArrayList<Integer>> allClusters;      //all search groups

        allClusters = new ArrayList<>();

        //convert tree search result to lists of names
        allKeys = nnGroups.keySet().iterator();
        while (allKeys.hasNext())
        {
            nextKey = allKeys.next();
            nextGroup = nnGroups.get(nextKey);
            allClusters.add(nextGroup);
        }

        return allClusters;
    }

    /**
     * Get the categories list updated with the new rows during training.
     * @return the value of categories.
     */
    public ArrayList<String> getCategories() {
        return categories;
    }

    /**
     * Get the dataset updated with new rows during training.
     * @return the value of dataRows.
     */
    public ArrayList<double[]> getTrainRows() {
        return trainRows;
    }
}
