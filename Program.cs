using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SpamFilter
{
    public class SpamInput
    {
        [LoadColumn(0)] public int ID { get; set; }
        [LoadColumn(1)] public string Subject { get; set; }
        [LoadColumn(2)] public string Message { get; set; }
        [LoadColumn(3)] public string RawLabel { get; set; }
        [LoadColumn(4)] public string Date { get; set; }

    }
    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsSpam { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    public class FromLabel
    {
        public string RawLabel { get; set; }
    }

    public class ToLabel
    {
        public bool Label { get; set; }
    }

    public class RandomForest
    {
        private string dataPath;
        private float testFraction;
        private int numberOfLeaves;
        private int numberOfTrees;

        public RandomForest(string dataPath, float testFraction, int numberOfLeaves, int numberOfTrees)
        {
            this.dataPath = dataPath;
            this.testFraction = testFraction;
            this.numberOfLeaves = numberOfLeaves;
            this.numberOfTrees = numberOfTrees;
        }

        public void Grow()
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<SpamInput>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var partitions = context.Data.TrainTestSplit(data,testFraction: testFraction);

            var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>(
                mapAction: (input, output) => { output.Label = input.RawLabel == "spam"; }, contractName: "pipeline")
                .Append(context.Transforms.Text.FeaturizeText(
                    outputColumnName: "Message",
                    inputColumnName: nameof(SpamInput.Message)))
                .Append(context.Transforms.Text.FeaturizeText(
                    outputColumnName: "Subject",
                    inputColumnName: nameof(SpamInput.Subject)))
                .Append(context.Transforms.Concatenate("Features", "Message", "Subject"))
                .Append(context.Transforms.NormalizeMinMax("Features", "Features"))
                .Append(context.BinaryClassification.Trainers.FastForest(numberOfLeaves: numberOfLeaves, numberOfTrees: numberOfTrees))                
                .AppendCacheCheckpoint(context);

            var trainedModel = pipeline.Fit(partitions.TrainSet);
            var testSetTransform = trainedModel.Transform(partitions.TestSet);
            var modelMetrics = context.BinaryClassification.EvaluateNonCalibrated(testSetTransform);

            Console.WriteLine($"Forest: ({numberOfLeaves}-{numberOfTrees})");
            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy}{Environment.NewLine}" +
                              $"F1 Score: {modelMetrics.F1Score}{Environment.NewLine}" +
                              $"Positive Precision: {modelMetrics.PositivePrecision}{Environment.NewLine}" +
                              $"Negative Precision: {modelMetrics.NegativePrecision}{Environment.NewLine}" +
                              $"Positive Recall: {modelMetrics.PositiveRecall}{Environment.NewLine}" +
                              $"Negative Recall: {modelMetrics.NegativeRecall}{Environment.NewLine}");
            Console.WriteLine("------------------------------------------------------------------------------------------------------");
        }

    }

    class Program
    {
        private static string dataPath = @"C:\Users\UKeselman\source\repos\SpamFilter\SpamFilter\enron_spam_data.csv";
        static void Main(string[] args)
        {
            var forests = new List<RandomForest>
            {
                new RandomForest(dataPath, 0.2f, 2, 5),
                new RandomForest(dataPath, 0.2f, 5, 10),
                new RandomForest(dataPath, 0.2f, 10, 20),
            };

            forests.ForEach(f => f.Grow());

            Console.ReadLine();
        }

    }
}
