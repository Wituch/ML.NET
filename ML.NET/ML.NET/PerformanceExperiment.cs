using Microsoft.ML;
using ML.NET.Models;
using System.Diagnostics;

namespace ML.NET
{
    public class PerformanceExperiment
    {
        private const string filePath = @"C:\CriteoSearchData";
        public static void ConductExperiment()
        {
            //Ładowanie danych
            var mlContext = new MLContext(42);
            var data = mlContext.Data.LoadFromTextFile<CriteoData>(filePath, separatorChar: '\t', hasHeader: true);

            //Pobranie 10% zbioru i podział na zbiór treningowi i testowy
            data = mlContext.Data.TakeRows(data, 1599563);
            var splitDataView = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            //Transformacja danych
            var map = new Dictionary<float, bool> {{ 1.0f, true }, { 0.0f, false }};
            var pipeline = mlContext.Transforms.Conversion.MapValue("Label", map)
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("Timestamp"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductAgeGroup"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("DeviceType"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("AudienceId"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductGender"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductBrand"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductCategory"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductCountry"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductId"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("ProductTitle"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("PartnerId"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("UserId"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("X1"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("X2"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("X3"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("X4"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("X5"))
                                                          .Append(mlContext.Transforms.Text.FeaturizeText("X6"))
                                                          .Append(mlContext.Transforms.Concatenate("Features", "Timestamp", "ClicksPerWeek", "ProductAgeGroup", "DeviceType", 
                                                          "AudienceId","ProductGender", "ProductBrand", "ProductCategory", "ProductCountry", "ProductId", "ProductTitle", 
                                                          "PartnerId", "UserId","X1", "X2", "X3", "X4", "X5", "X6"))
                                                          .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", 
                                                          featureColumnName: "Features"));
            //Trenowanie modelu
            var model = pipeline.Fit(splitDataView.TrainSet);

            //Ewaluacja wyników
            var testDataView = model.Transform(splitDataView.TestSet);
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            var cvResults = mlContext.BinaryClassification.CrossValidate(testDataView, mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(), numberOfFolds: 5);

            stopwatch.Stop();
            var time = stopwatch.Elapsed;

        }
    }
}
