/*
 * StepNN_V2.java
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
import org.ai_heuristic.util.AiHeuristicConst;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;


/**
 * New version that adds new points as steps to outliers.
 * This version selects any point from the same category, not just the closest one,
 * so long as the new points added are not then closer to other category points.
 */
public class StepNN_V2 extends StepNN {

    /**
     * Create a new instance of StepNN_V2.
     * @param evalType the type of evaluation metric.
     * @param ds the dataset.
     * @throws Exception any error.
     */
    public StepNN_V2(String evalType, Dataset ds) throws Exception {

        super(evalType, ds);
    }

    /**
     * Calculate the closest data point from each point to the other and use this to cluster.
     * @return the cluster indexes from the {@code datasets} list, as {@code ArrayList} lists.
     * You can use {@code ((Integer)ReplySet.getValue()).intValue()} to retrieve the cluster values.
     * @throws Exception any error.
     */
    public ReplySet evaluate() throws Exception {
        int i, j, k;
        boolean isOwn;                                          //true if own category
        boolean isCloser;                                       //true if closer point
        double distance;                                        //distance
        double closeDist, myDist;                               //distances
        double[] dataRow, otherRow;                             //data rows
        double[] newRow, firstRow;                              //new data point
        String myLabel;                                         //categories
        ArrayList<Integer> indexList;                           //index list
        ArrayList<Integer> posList;                             //index list
        ArrayList<Integer> cluster;                             //a cluster
        HashMap<Integer, ArrayList<Integer>> clusters;          //list of clusters

        clusters = new HashMap<>();

        //generate all distances from one point to every other point
        for (i = 0; i < dataRows.size(); i++)
        {
            dataRow = dataRows.get(i);

            myDistances = predict(i, dataRow, false);
            distOrdered = new ArrayList(myDistances.keySet());
            Collections.sort(distOrdered);

            myLabel = categories.get(i);
            cluster = new ArrayList<>();
            cluster.add(i);
            clusters.put(i, cluster);
            isOwn = false;

            //determine other closest
            closeDist = distOrdered.get(0);
            indexList = myDistances.get(closeDist);
            posList = matchList(myLabel, indexList, categories);
            cluster.addAll(indexList);

            if (!isMatch(indexList, posList)) {
                isCloser = false;
                firstRow = null;

                for (j = 1; (!isCloser || !isOwn) && (j < distOrdered.size()); j++) {
                    myDist = distOrdered.get(j);
                    indexList = myDistances.get(myDist);
                    posList = matchList(myLabel, indexList, categories);

                    if (isMatch(indexList, posList)) {
                        otherRow = trainRows.get(posList.get(0));
                        isOwn = true;

                        //shift incorrectly classified row by a small amount in opposite direction to average row
                        newRow = new double[otherRow.length];
                        for (k = 0; k < otherRow.length; k++) {
                            newRow[k] = otherRow[k];
                        }
                        for (k = 0; k < newRow.length; k++) {
                            newRow[k] += ((dataRow[k] - newRow[k]) / 2.0);
                        }

                        //If cut-off by a point from different category, then maybe an outlier
                        // 1. When add new point, measure distance to current point from it.
                        // 2. New point should not be closer to any point from different category then this distance.
                        // 3. These are likely to be points further up the list because closer but different category.
                        // 4. So when select point, measure distances from all above it as well.
                        distance = evaluate(dataRow, newRow, evalType);

                        if (firstRow == null) {
                            firstRow = newRow;
                        }

                        //if not closer, then need to try with a different point
                        //otherwise can save and move onto next point
                        isCloser = isCloser(j, distance, newRow, myLabel);
                        if (isCloser) {
                            trainRows.add(newRow);
                            categories.add(myLabel);
                        }
                    }
                }

                //add first point anyway if none are closer
                if (!isCloser && firstRow != null) {
                    trainRows.add(firstRow);
                    categories.add(myLabel);
                }
            }
        }

        return ReplySet.toSet(toClusters(clusters));
    }


    /**
     * Determine if new point is closer to any other category.
     * @param ordIndex index in ordered distance list for own point in question.
     * @param minDist the distance to be closer than.
     * @param dataRow the data row to measure.
     * @param dataLabel label for own point in question.
     * @return true if the row point is closer to some other category point.
     * @throws Exception
     */
    private boolean isCloser(int ordIndex, double minDist, double[] dataRow, String dataLabel)
            throws Exception {
        int i;
        boolean isCloser;                               //true if closer
        double distance, myDist;                        //distance
        double[] otherRow;                              //data rows
        ArrayList<Integer> indexList;                   //index list
        ArrayList<Integer> posList;                     //index list

        isCloser = true;
        for (i = 0; isCloser && (i < ordIndex); i++) {
            myDist = distOrdered.get(i);
            indexList = myDistances.get(myDist);
            posList = matchList(dataLabel, indexList, categories);

            if (isMatch(indexList, posList)) {
                otherRow = trainRows.get(posList.get(0));
                distance = evaluate(dataRow, otherRow, evalType);
                isCloser = (minDist < distance);
            }
        }

        return isCloser;
    }
}
