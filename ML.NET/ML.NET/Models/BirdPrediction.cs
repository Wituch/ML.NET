using Microsoft.ML.Data;

namespace ML.NET.Models
{
    public class BirdPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsMale { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
