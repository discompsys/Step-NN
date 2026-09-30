/*
 * TestStepNN.java
 *
 * Created on 14 October 2025.
 *
 * Version 1.0
 *
 * Copyright (c) Kieran Greer
 */

package org.dcs.stepnn;


import org.ai_heuristic.data.Dataset;
import org.ai_heuristic.data.Normalise;
import org.ai_heuristic.eval.metric.ReplySet;
import org.ai_heuristic.eval.result.ClusterResult;
import org.ai_heuristic.util.AiHeuristicConst;
import org.jlog2.CustomLoggerFactory;
import org.jlog2.LoggerFactory;

import java.util.ArrayList;


/** Test the new Step NN algorithm */
public class TestStepNN {

    /**
     * Create a new instance of TestStepNN.
     */
    public TestStepNN()
    {
        runTest();
    }

    /**
     * Run some tests.
     */
    private void runTest()
    {
        int testLoop, stepLoop;
        int catStartEnd;                            //cat at start or end
        long timeSnn;                               //snn timing
        long timeSnnTr1, timeSnnTotalTr1, timeSnnTotalTst1;     //snn timing
        long timeSnnTr2, timeSnnTotalTr2, timeSnnTotalTst2;     //snn timing
        double stepTotalV1, stepTotalV2;              //total wrong
        String trnFilePath, tstFilePath;            //the data file paths
        Dataset dataset;                            //test dataset
        Dataset[] trainTestSNN;                     //train and test datasets
        StepNN_V1 snnV1, snnTestV1;                 //step nn
        StepNN_V2 snnV2, snnTestV2;                 //step nn version 1
        ClusterResult notMatch;                     //errors
        ReplySet reply;                             //evaluation result

        try
        {
            //only train set
            trnFilePath = "C:/Data/dataset.txt";
            tstFilePath = null;

            //or separate test set
            //tstFilePath = "C:/Data/dataset2.txt";

            //number of separate tests, to average over
            final int NEWTESTRUNS = 50;

            //number of iterations in a single test
            final int STEPITER = 15;

            //make this 1 if the output category is at the start,
            // or 2 if it is at the end
            catStartEnd = 2;

            //NOTE use dataset with no meta at the start
            // output category set by catStartEnd = 1 for first col, or 2 for last col
            stepTotalV1 = 0;
            stepTotalV2 = 0;
            timeSnnTotalTr1 = 0;
            timeSnnTotalTst1 = 0;
            timeSnnTotalTr2 = 0;
            timeSnnTotalTst2 = 0;
            trainTestSNN = null;

            for (testLoop = 0; testLoop < NEWTESTRUNS; testLoop++) {
                //test parameters
                dataset = new Dataset(trnFilePath, ",", catStartEnd);
                Normalise.normaliseSeparate(dataset);
                timeSnnTr1 = 0;
                timeSnnTr2 = 0;

                //can split into train and test
                trainTestSNN = dataset.trainTest(25);

                //or keep as same dataset
                //trainTest[0] = dataset;
                //trainTest[1] = dataset;
                //or can have completely different test set, that is read later

                snnV1 = new StepNN_V1(AiHeuristicConst.EUCLIDEANFUNCTION, trainTestSNN[0]);
                snnV1.dataToFile("C:\\Users\\DCS\\Documents\\My Docs\\stepNNV1_Orig.txt");
                timeSnn = System.currentTimeMillis();
                reply = snnV1.evaluate();
                timeSnnTr1 += System.currentTimeMillis() - timeSnn;

                snnV2 = new StepNN_V2(AiHeuristicConst.EUCLIDEANFUNCTION, trainTestSNN[0]);
                snnV2.dataToFile("C:\\Users\\DCS\\Documents\\My Docs\\stepNNV2_Orig.txt");
                timeSnn = System.currentTimeMillis();
                snnV2.evaluate();
                timeSnnTr2 += System.currentTimeMillis() - timeSnn;

                if (testLoop == NEWTESTRUNS-1) {
                    System.out.println("\nNew Step NN V1 dataset size: " + snnV1.getTrainRows().size());
                    System.out.println("New Step NN V2 dataset size: " + snnV2.getTrainRows().size());

                    /*notMatch = snnV1.notMatch((ArrayList<ArrayList<String>>) reply.getValue(),
                            snnV1.getCategories(), snnV1.getCategories());
                    System.out.println("\nStep NN V1 final Clusters that conflict:");
                    System.out.println(String.valueOf(notMatch.clusters));*/
                }

                //try again
                for (stepLoop = 0; stepLoop < STEPITER; stepLoop++) {
                    timeSnn = System.currentTimeMillis();
                    reply = snnV1.evaluate();
                    timeSnnTr1 += System.currentTimeMillis() - timeSnn;

                    timeSnn = System.currentTimeMillis();
                    snnV2.evaluate();
                    timeSnnTr2 += System.currentTimeMillis() - timeSnn;

                    if (stepLoop == STEPITER -1) {
                        System.out.println("\nNew Step NN V1 dataset size: " + snnV1.getTrainRows().size());
                        System.out.println("New Step NN V2 dataset size: " + snnV2.getTrainRows().size());

                        /*notMatch = snnV1.notMatch((ArrayList<ArrayList<String>>) reply.getValue(),
                                snnV1.getCategories(), snnV1.getCategories());
                        System.out.println("\nStep NN V1 loop Clusters that conflict:");
                        System.out.println(String.valueOf(notMatch.clusters));*/
                    }
                }

                //factor train time by STEPITER
                timeSnnTotalTr1 += timeSnnTr1;
                timeSnnTotalTr2 += timeSnnTr2;

                //if separate test file declared, then use it
                if (tstFilePath != null) {
                    dataset = new Dataset(tstFilePath, ",", catStartEnd);
                    Normalise.normaliseSeparate(dataset, false);
                    trainTestSNN[1] = dataset;
                }

                snnTestV1 = new StepNN_V1(AiHeuristicConst.EUCLIDEANFUNCTION, trainTestSNN[1]);
                timeSnn = System.currentTimeMillis();
                reply = snnTestV1.testOnly(trainTestSNN[1].getDataset(), snnV1.getTrainRows(),
                        snnV1.getCategories());
                timeSnnTotalTst1 += System.currentTimeMillis() - timeSnn;

                notMatch = snnV1.notMatch((ArrayList<ArrayList<Integer>>) reply.getValue(),
                        snnV1.getCategories(), snnTestV1.getCategories());
                stepTotalV1 += notMatch.clusters.size();

                System.out.println("\nTest STEP V1 Result: Clusters that conflict:");
                System.out.println(String.valueOf(notMatch.clusters.size()));


                snnTestV2 = new StepNN_V2(AiHeuristicConst.EUCLIDEANFUNCTION, trainTestSNN[1]);
                timeSnn = System.currentTimeMillis();
                reply = snnTestV2.testOnly(trainTestSNN[1].getDataset(), snnV2.getTrainRows(),
                        snnV2.getCategories());
                timeSnnTotalTst2 += System.currentTimeMillis() - timeSnn;

                notMatch = snnV2.notMatch((ArrayList<ArrayList<Integer>>) reply.getValue(),
                        snnV2.getCategories(), snnTestV2.getCategories());
                stepTotalV2 += notMatch.clusters.size();

                System.out.println("\nTest STEP V2 Result: Clusters that conflict:");
                System.out.println(String.valueOf(notMatch.clusters.size()));


                //test using basic knn with ksize of 3
                dataset = new Dataset(trnFilePath, ",", catStartEnd);
                Normalise.normaliseSeparate(dataset);
            }

            stepTotalV1 /= NEWTESTRUNS;
            System.out.println("\n\nAverage Test STEP V1 Error: " + stepTotalV1);
            stepTotalV1 = (1.0 - (stepTotalV1 / trainTestSNN[1].getDataSize())) * 100.0;
            System.out.println("Which is " + stepTotalV1 + " % correct.");

            timeSnnTotalTr1 /= NEWTESTRUNS;
            timeSnnTotalTst1 /= NEWTESTRUNS;
            System.out.println("\nAverage Train STEP V1 time: " + timeSnnTotalTr1);
            System.out.println("Average Test STEP V1 time: " + timeSnnTotalTst1);

            stepTotalV2 /= NEWTESTRUNS;
            System.out.println("\nAverage Test STEP V2 Error: " + stepTotalV2);
            stepTotalV2 = (1.0 - (stepTotalV2 / trainTestSNN[1].getDataSize())) * 100.0;
            System.out.println("Which is " + stepTotalV2 + " % correct.");

            timeSnnTotalTr2 /= NEWTESTRUNS;
            timeSnnTotalTst2 /= NEWTESTRUNS;
            System.out.println("\nAverage Train STEP V2 time: " + timeSnnTotalTr2);
            System.out.println("Average Test STEP V2 time: " + timeSnnTotalTst2);
        }
        catch (Exception ex)
        {
            ex.printStackTrace(System.out);
            System.exit(0);
        }
    }

    /**
     * Main method to initiate the tests.
     * @param args command line arguments.
     */
    public static void main(String[] args)
    {
        try
        {
            LoggerFactory.setLoggerFactory(new CustomLoggerFactory());
            new TestStepNN();
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
