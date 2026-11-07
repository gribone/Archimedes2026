# Temporary Google Colab workflow

Use the included `notebooks/ForceCRM_Colab.ipynb`. Colab runs on Google's servers, so it is **for temporary experimentation, not the final challenge deployment**. This workflow runs open models inside the allocated VM rather than calling a hosted AI inference API. Uploading data still transfers it to Google: only use data you are authorized to upload.

1. Create a ZIP with `forcecrm/`, `scripts/`, `requirements.txt`, `requirements-ai.txt`, and the three supplied CSVs. Put these at the ZIP root (a single enclosing folder is also supported). Include your customer workbook if permitted. Exclude `.venv/`, `models/`, `.cache/` and `outputs/`. A source ZIP without competition data is provided as `forcecrm-source.zip`; upload the three CSVs separately in the notebook when using that ZIP.
2. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and open `ForceCRM_Colab.ipynb`.
3. Choose **Runtime → Change runtime type → T4 GPU** if available. CPU works but is slower. Free resources are not guaranteed.
4. Leave `DEMO = False` in the first code cell for your real workbook, or set it to `True` for a synthetic demonstration. Run the cells in order. Upload the ZIP, then supply missing CSVs and the customer `.xlsx` when prompted. Keep the expected CSV filenames unchanged.
5. The installation cell installs the pinned Python dependencies. The next cells install official Ollama into the temporary VM, start its loopback server, pull `ministral-3:3b`, and download E5-small. Downloads are several GB and can take minutes. They happen before analysis.
6. Run the analysis cell. With `DEMO = False`, E5 and Ministral process the uploaded workbook. With `DEMO = True`, the notebook skips model provisioning and uses synthetic customers and lexical features. Results appear directly in notebook cells.
7. Download the results ZIP before the runtime expires. VM files are temporary. The notebook displays profile text, validation metrics and recommendations, and exports the same fixed-format files as the local application.

The notebook deliberately does not create a public tunnel or remote-host the application. Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) restricts bypassing the notebook UI to interact primarily through a web UI on free managed runtimes without a positive compute balance. Use notebook cells for Colab experimentation. To test the full web interface, return to your laptop:

```powershell
python run.py --setup-models
python run.py
```

Then open http://127.0.0.1:8000. No Colab connection is required for the final implementation.
