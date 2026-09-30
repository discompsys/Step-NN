/*
 * StepNN_V1.java
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
 * This version selects only the closest point from the same category.
 */
public class StepNN_V1 extends StepNN {

    /**
     * Create a new instance of StepNN_V1.
     * @param evalType the type of evaluation metric.
     * @param ds the dataset.
     * @throws Exception any error.
     */
    public StepNN_V1(String evalType, Dataset ds) throws Exception {

        super(evalType, ds);
    }

    /**
     * Calculate the closest data points from each point to the other and use this to cluster.
     * @return the cluster indexes from the {@code datasets} list, as {@code ArrayList} lists.
     * You can use {@code ((Integer)ReplySet.getValue()).intValue()} to retrieve the cluster values.
     * @throws Exception any error.
     */
    public ReplySet evaluate() throws Exception {
        int i, j, k;
        boolean isOwn;
        double myDist;                                          //distances
        double[] dataRow, otherRow;                             //data rows
        double[] newRow;                                        //new data point
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

            //determine other closest
            myDist = distOrdered.get(0);
            indexList = myDistances.get(myDist);
            posList = matchList(myLabel, indexList, categories);
            cluster.addAll(posList);

            if (!isMatch(indexList, posList)) {
                isOwn = false;
                for (j = 1; !isOwn && (j < distOrdered.size()); j++) {
                    myDist = distOrdered.get(j);
                    indexList = myDistances.get(myDist);
                    posList = matchList(myLabel, indexList, categories);

                    if (isMatch(indexList, posList)) {
                        isOwn = true;

                        //if not first row then try to add link point
                        otherRow = trainRows.get(posList.get(0));

                        //shift incorrectly classified row by a small amount in opposite direction to average row
                        newRow = new double[otherRow.length];
                        for (k = 0; k < otherRow.length; k++) {
                            newRow[k] = otherRow[k];
                        }
                        for (k = 0; k < newRow.length; k++) {
                            newRow[k] += ((dataRow[k] - newRow[k]) / 2.0);
                        }

                        trainRows.add(newRow);
                        categories.add(myLabel);
                    }
                }
            }
        }

        return ReplySet.toSet(toClusters(clusters));
    }
}
