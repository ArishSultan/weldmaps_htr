For PDF image extraction use following command

```bash
python pdf_utils.py <input-pdf> <pages-count>
```

Use lesser pages-count for testing purpose

--------------

For applying OCR on the extracted images use following command
```bash
python main.py <input-image>
```

NOTE: only the images extracted by pdf_utils having name *_1.png can be used as input to main.py