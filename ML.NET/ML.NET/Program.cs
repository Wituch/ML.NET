using Microsoft.ML;
using Microsoft.ML.AutoML;
using ML.NET;
using ML.NET.Models;

class Program
{
    private const string filePath = @"C:\AVONET_Raw_Data.csv";
    static void Main(string[] args)
    {
        //Ładowanie danych
        var mlContext = new MLContext(42);
        var data = mlContext.Data.LoadFromTextFile<Bird>(filePath, separatorChar: ',', hasHeader: true);
        var splitDataView = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        //Transformacja danych
        var map = new Dictionary<string, bool> { { "M", true }, { "F", false } };
        var pipeline = mlContext.Transforms.Conversion.MapValue("Label", map)
                                                      .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Species", outputColumnName: "SpeciesFeaturized"))
                                                      .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Country", outputColumnName: "CountryFeaturized"))
                                                      .Append(mlContext.Transforms.Concatenate("Features", "SpeciesFeaturized", "CountryFeaturized", "BeakLengthCulmen",
                                                      "BeakLengthNares", "BeakWidth", "BeakDepth", "TarsusLength", "WingLength", "KippsDistance", "SecondaryLength", 
                                                      "HandWingIndex", "TailLength"))
                                                      .Append(mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"));

        //Trenowanie modelu
        var model = pipeline.Fit(splitDataView.TrainSet);

        //Sprawdzenie klasyfikacji
        var testBird = new Bird 
        { 
            Species = "Bubo africanus", 
            Country = "Kenya",
            BeakLengthCulmen = 38.9f,
            BeakLengthNares = 19.1f,
            BeakWidth = 10.7f,
            BeakDepth = 17.2f,
            TarsusLength = 53.5f,
            WingLength = 320f,
            KippsDistance = 98.2f,
            SecondaryLength = 221.8f,
            HandWingIndex = 30.7f,
            TailLength = 162f

        };
        var predictedBirdSex = mlContext.Model.CreatePredictionEngine<Bird, BirdPrediction>(model).Predict(testBird).IsMale;


        //Ewaluacja wyników
        var testDataView = model.Transform(splitDataView.TestSet);
        var cvResults = mlContext.BinaryClassification.CrossValidate(testDataView, mlContext.BinaryClassification.Trainers.LightGbm(), numberOfFolds: 5);
        
        //Obliczenie metryk
        var accuracy = cvResults.Select(f => f.Metrics.Accuracy).Average();
        var f1 = cvResults.Select(f => f.Metrics.F1Score).Average();
        var auroc = cvResults.Select(f => f.Metrics.AreaUnderRocCurve).Average();
        var precision = cvResults.Select(f => f.Metrics.PositivePrecision).Average();
        var recall = cvResults.Select(f => f.Metrics.PositiveRecall).Average();

        //Auto ML
        //Ustawienia eksperymentu
        var experimentSettings = new BinaryExperimentSettings();
        experimentSettings.MaxExperimentTimeInSeconds = 900;
        experimentSettings.OptimizingMetric = BinaryClassificationMetric.Accuracy;
        experimentSettings.CacheDirectoryName = null;

        //Transformacja danych
        var transformedData = mlContext.Transforms.Conversion.MapValue("Label", new Dictionary<string, bool> { { "M", true }, { "F", false } }).Fit(data).Transform(data);

        //Przeprowadzenie eksperymentu
        var experiment = mlContext.Auto().CreateBinaryClassificationExperiment(experimentSettings);
        var experimentResult = experiment.Execute(transformedData);

        //Ewaluacja danych
        var metrics = experimentResult.BestRun.ValidationMetrics;
        var accuracyAuto = metrics.Accuracy;
        var f1Auto = metrics.F1Score;
        var aurocAuto = metrics.AreaUnderRocCurve;
        var precisionAuto = metrics.PositivePrecision;
        var recallAuto = metrics.PositiveRecall;

        //Sprawdzenie wydajności
        PerformanceExperiment.ConductExperiment();
    }
}