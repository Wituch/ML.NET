using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NET.Models
{
    public class CriteoData
    {
        [LoadColumn(0), ColumnName("Label")]
        public float Sale { get; set; }

        [LoadColumn(3)]
        public string Timestamp { get; set; }

        [LoadColumn(2)]
        public float ClicksPerWeek { get; set; }

        [LoadColumn(6)]
        public string ProductAgeGroup { get; set; }

        [LoadColumn(7)]
        public string DeviceType { get; set; }

        [LoadColumn(8)]
        public string AudienceId { get; set; }

        [LoadColumn(9)]
        public string ProductGender { get; set; }

        [LoadColumn(10)]
        public string ProductBrand { get; set; }

        [LoadColumn(11)]
        public string ProductCategory { get; set; }

        [LoadColumn(12)]
        public string ProductCountry { get; set; }

        [LoadColumn(13)]
        public string ProductId { get; set; }

        [LoadColumn(14)]
        public string ProductTitle { get; set; }

        [LoadColumn(15)]
        public string PartnerId { get; set; }

        [LoadColumn(16)]
        public string UserId { get; set; }

        [LoadColumn(17)]
        public string X1 { get; set; }

        [LoadColumn(18)]
        public string X2 { get; set; }

        [LoadColumn(19)]
        public string X3 { get; set; }

        [LoadColumn(20)]
        public string X4 { get; set; }

        [LoadColumn(21)]
        public string X5 { get; set; }

        [LoadColumn(22)]
        public string X6 { get; set; }
    }
}
