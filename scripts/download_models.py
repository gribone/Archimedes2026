"""The only embedding download entrypoint. Never imported by the web runtime."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='intfloat/multilingual-e5-small')
    parser.add_argument('--destination', default='models/e5-small')
    args = parser.parse_args()
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(args.model).sha
    path = Path(args.destination)
    snapshot_download(args.model, revision=revision, local_dir=path,
                      allow_patterns=['*.json', '*.safetensors', '*.model', '1_Pooling/*', 'README.md', 'LICENSE*'],
                      ignore_patterns=['onnx/*', 'openvino/*'])
    (path / 'forcecrm-model.json').write_text(json.dumps({'model': args.model, 'revision': revision}, indent=2), 'utf-8')
    print(f'Prepared {args.model} at revision {revision}: {path}')


if __name__ == '__main__':
    main()
