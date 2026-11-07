import argparse
from pathlib import Path
from forcecrm.data import load_universe, read_portfolio, demo_portfolio
from forcecrm.pipeline import run_analysis, export_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio', type=Path)
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('outputs/cli'))
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--model', default='ministral-3:3b')
    parser.add_argument('--bootstrap', type=int, default=20)
    args = parser.parse_args()
    if not args.demo and not args.portfolio:
        parser.error('--portfolio is required unless --demo is selected')
    if args.bootstrap < 2:
        parser.error('--bootstrap must be at least 2')
    c, a, _, _ = load_universe(args.data)
    portfolio = demo_portfolio(c, a) if args.demo else read_portfolio(args.portfolio, c)
    result = run_analysis(args.data, portfolio, demo=args.demo, model=args.model,
                          bootstrap=args.bootstrap, progress=lambda m: print(m, flush=True))
    export_result(result, args.output)
    print(f'Saved {len(result["profiles"])} profiles and {len(result["lookalikes"])} candidates to {args.output}')


if __name__ == '__main__':
    main()
