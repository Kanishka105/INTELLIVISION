from paddleocr import PaddleOCR


class OCRReader:

    def __init__(self):

        self.ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )


    def read(self, image):

        if image is None:

            return {
                "text": "",
                "confidence": 0.0
            }


        results = self.ocr.predict(
            image
        )


        texts = []
        scores = []


        for result in results:

            data = result.json

            if isinstance(data, dict):

                data = data.get(
                    "res",
                    data
                )


            rec_texts = data.get(
                "rec_texts",
                []
            )

            rec_scores = data.get(
                "rec_scores",
                []
            )


            texts.extend(
                rec_texts
            )

            scores.extend(
                rec_scores
            )


        if not texts:

            return {
                "text": "",
                "confidence": 0.0
            }


        text = " ".join(
            texts
        )


        confidence = (
            max(scores)
            if scores
            else 0.0
        )


        return {

            "text": text,

            "confidence":
                float(confidence)
        }


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    print(
        "✅ OCRReader loaded"
    )

    reader = OCRReader()

    print(
        "✅ PaddleOCR initialized"
    )