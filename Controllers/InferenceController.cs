using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ONNXAPI.Controllers
{
    [ApiController]
    [Route("/score")]
    public class InferenceController : ControllerBase
    {
        private InferenceSession _session;

        public InferenceController(InferenceSession session)
        {
            _session = session;
        }

        [HttpPost]
        //public ActionResult Score(Data data)
        //{
        //    var result = _session.Run(new List<NamedOnnxValue>
        //    {
        //        NamedOnnxValue.CreateFromTensor("input", data.AsTensor())
        //    });
        //    Tensor<float> output = result.First().AsTensor<float>();
        //    int predictionIndex = output.ArgMax();
        //    string[] categories = new string[] { "B", "H", "W" };
        //    var prediction = new Prediction { PredictedValue = categories[predictionIndex] };
        //    result.Dispose();
        //    return Ok(prediction);
        //}

        [HttpPost]
        public ActionResult Score(Data data)
        {
            var result = _session.Run(new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", data.AsTensor())
                });

            Tensor<string> output_label = result.First().AsTensor<string>();


            var categories = new[] { "B", "H", "W" };
            int predictionIndex = Array.IndexOf(output_label.ToArray(), output_label.Max());
            var prediction = new Prediction { PredictedValue = categories[predictionIndex] };
            return Ok(prediction);

            //string[] categories = { "B", "H", "W" };

            //// Get the index of the category with the highest score
            //int predictionIndex = Array.IndexOf(output.ToArray(), output.Max());

            //// Create a prediction object with the predicted category
            //var prediction = new Prediction { PredictedValue = categories[predictionIndex] };

            //result.Dispose();
            //return Ok(prediction);
        }
    }
}