using Microsoft.ML.Data;

namespace ML.NET.Models
{
    public class Bird
    {
        [LoadColumn(8), ColumnName("Label")]
        public string Sex { get; set; }

        [LoadColumn(1)]
        public string Species { get; set; }
        [LoadColumn(11)]
        public string Country { get; set; }
        [LoadColumn(13)]
        public float BeakLengthCulmen { get; set; }
        [LoadColumn(14)]
        public float BeakLengthNares { get; set; }
        [LoadColumn(15)]
        public float BeakWidth { get; set; }
        [LoadColumn(16)]
        public float BeakDepth { get; set; }
        [LoadColumn(17)]
        public float TarsusLength { get; set; }
        [LoadColumn(18)]
        public float WingLength { get; set; }
        [LoadColumn(19)]
        public float KippsDistance { get; set; }
        [LoadColumn(20)]
        public float SecondaryLength { get; set; }
        [LoadColumn(21)]
        public float HandWingIndex { get; set; }
        [LoadColumn(22)]
        public float TailLength { get; set; }
    }
}
