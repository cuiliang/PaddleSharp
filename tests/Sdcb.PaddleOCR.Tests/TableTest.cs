using OpenCvSharp;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.LocalV3;

namespace Sdcb.PaddleOCR.Tests;

public class TableTest
{
    [Theory]
    //[InlineData("en_ppstructure_mobile_v2.0_SLANet")]
    [InlineData("ch_ppstructure_mobile_v2.0_SLANet")]
    public void LocalV3TableTest(string modelName)
    {
        using PaddleOcrTableRecognizer tableRec = new(LocalTableRecognitionModel.All.Single(x => x.Name == modelName));
        
        //using Mat src = Cv2.ImRead("samples/table.jpg");
        //using Mat src = Cv2.ImRead("E:\\Drive\\AI\\表格识别原图\\表格识别原图\\d_9.jpg");

        string path = "E:\\Drive\\AI\\表格识别原图\\表格识别原图\\d_9.jpg";

        using Mat src = Cv2.ImRead(path);

        TableDetectionResult result = tableRec.Run(src);
        Assert.NotNull(result);
        Assert.NotEmpty(result.StructureBoxes);
        Assert.NotEmpty(result.HtmlTags);
        Assert.True(result.Score > 0.9f);
        using Mat visualized = result.Visualize(src, Scalar.Red);
        Cv2.ImShow("result", visualized);
        //Cv2.WaitKey();


        using PaddleOcrAll all = new(LocalFullModels.ChineseV3);
        all.Detector.UnclipRatio = 1.2f;
        PaddleOcrResult ocrResult = all.Run(src);
        using Mat visualizedOcr = src.Clone();
        foreach (var region in ocrResult.Regions)
        {
            var rect = region.Rect;
            DrawRotatedRect(visualizedOcr, rect, Scalar.Green, 1);
        }
        Cv2.ImShow("ocr", visualizedOcr);
        Cv2.WaitKey();

        Cv2.ImWrite(path + ".table-visualized.jpg", visualized);
    }

    void DrawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness)
    {
        // 获取RotatedRect的四个顶点
        Point2f[] vertices = rotatedRect.Points();

        // 将顶点转换为整数坐标（Point类型）
        Point[] points = Array.ConvertAll(vertices, point => new Point((int)point.X, (int)point.Y));

        // 绘制四条线段来表示RotatedRect
        for (int i = 0; i < 4; i++)
        {
            Cv2.Line(image, points[i], points[(i + 1) % 4], color, thickness);
        }
    }

    [Theory]
    [InlineData("en_ppstructure_mobile_v2.0_SLANet", "<table><thead><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr></thead><tbody><tr>")]
    [InlineData("ch_ppstructure_mobile_v2.0_SLANet", "<table><tbody><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr><tr><td>SegLink[26]")]
    public void LocalV3TableRebuild(string modelName, string expectedHtmlStart)
    {
        using PaddleOcrTableRecognizer tableRec = new(LocalTableRecognitionModel.All.Single(x => x.Name == modelName));
        using Mat src = Cv2.ImRead("samples/table.jpg");
        TableDetectionResult tableResult = tableRec.Run(src);

        using PaddleOcrAll all = new(LocalFullModels.ChineseV3);
        all.Detector.UnclipRatio = 1.2f;
        PaddleOcrResult ocrResult = all.Run(src);

        string html = tableResult.RebuildTable(ocrResult);
        Assert.StartsWith(expectedHtmlStart, html);
    }

    [Theory]
    [InlineData("en_ppstructure_mobile_v2.0_SLANet", "<table><thead><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr></thead><tbody><tr>")]
    [InlineData("ch_ppstructure_mobile_v2.0_SLANet", "<table><tbody><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr><tr><td>SegLink[26]")]
    public async Task OnlineV3TableRebuild(string modelName, string expectedHtmlStart)
    {
        OnlineTableRecognitionModel tableOnlineModel = OnlineTableRecognitionModel.All.Single(x => x.Name == modelName);
        TableRecognitionModel tableModel = await tableOnlineModel.DownloadAsync();
        using PaddleOcrTableRecognizer tableRec = new(tableModel);
        using Mat src = Cv2.ImRead("samples/table.jpg");
        TableDetectionResult tableResult = tableRec.Run(src);

        using PaddleOcrAll all = new(LocalFullModels.ChineseV3);
        all.Detector.UnclipRatio = 1.2f;
        PaddleOcrResult ocrResult = all.Run(src);

        string html = tableResult.RebuildTable(ocrResult);
        Assert.StartsWith(expectedHtmlStart, html);
    }

    [Theory]
    [InlineData("en_ppstructure_mobile_v2.0_SLANet")]
    [InlineData("ch_ppstructure_mobile_v2.0_SLANet")]
    public async Task OnlineV3TableTest(string modelName)
    {
        OnlineTableRecognitionModel tableOnlineModel = OnlineTableRecognitionModel.All.Single(x => x.Name == modelName);
        TableRecognitionModel tableModel = await tableOnlineModel.DownloadAsync();
        using PaddleOcrTableRecognizer tableRec = new(tableModel);
        using Mat src = Cv2.ImRead("samples/table.jpg");
        TableDetectionResult result = tableRec.Run(src);
        Assert.NotNull(result);
        Assert.NotEmpty(result.StructureBoxes);
        Assert.NotEmpty(result.HtmlTags);
        Assert.True(result.Score > 0.9f);
        //using Mat visualized = result.Visualize(src, Scalar.LightGreen);
        //Cv2.ImWrite("table-visualized.jpg", visualized);
    }
}
