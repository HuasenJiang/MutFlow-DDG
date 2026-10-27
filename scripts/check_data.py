"""Check configured input locations without loading pickle/checkpoint objects."""
from __future__ import annotations

import argparse
from pathlib import Path

from mutflow_ddg.config import load_config


def required_paths(config, stage):
    if stage == 'pretrain':
        if not config.pretraining.data_path:
            raise ValueError('pretraining.data_path must be configured')
        yield 'pretraining.data_path', Path(config.pretraining.data_path), 'file'
        if config.pretraining.validation_path:
            yield 'pretraining.validation_path', Path(config.pretraining.validation_path), 'file'
        return
    if not config.data.train_path:
        raise ValueError('data.train_path must be configured')
    yield 'data.train_path', Path(config.data.train_path), 'file'
    if config.data.validation_path:
        yield 'data.validation_path', Path(config.data.validation_path), 'file'
    if config.model.structure_checkpoint:
        yield 'model.structure_checkpoint', Path(config.model.structure_checkpoint), 'file'
    uses_esm = (config.model.branch_mode in {'both', 'sequence'} and
                config.data.sequence_feature_source in {'esm', 'esm_pssm'})
    if uses_esm:
        if not config.data.esm_cache_dir:
            raise ValueError('data.esm_cache_dir must be configured for ESM input')
        yield 'data.esm_cache_dir', Path(config.data.esm_cache_dir), 'directory'
    for test in config.data.external_tests:
        yield f'external.{test.name}.path', Path(test.path), 'file'
        if uses_esm:
            if not test.esm_cache_dir:
                raise ValueError(f'External ESM cache root is not configured: {test.name}')
            yield f'external.{test.name}.esm', Path(test.esm_cache_dir), 'directory'
        if test.manifest_path:
            yield f'external.{test.name}.manifest', Path(test.manifest_path), 'file'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['pretrain', 'train'], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    missing = 0
    for name, path, kind in required_paths(config, args.stage):
        valid = path.is_dir() if kind == 'directory' else path.is_file()
        print(f'{"OK" if valid else "MISSING"}: {name}: {path}')
        missing += int(not valid)
    if missing:
        parser.exit(1, f'{missing} required locations are absent; see docs/DATA.md.\n')
    print('Configured root locations exist. Per-record content/alignment has not been validated.')


if __name__ == '__main__':
    main()
