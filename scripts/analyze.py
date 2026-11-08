import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from forcecrm.data import load_universe, read_portfolio, demo_portfolio
from forcecrm.narrative import DEFAULT_LLM
from forcecrm.pipeline import run_analysis, export_result, make_runner, prepare_filters
from forcecrm.stages import STAGES, ORDER


def downstream(name):
    """`name` and every stage that depends on it, directly or not."""
    found = {name}
    for stage in ORDER:  # ORDER is topological
        if found & set(STAGES[stage].needs):
            found.add(stage)
    return found


def main():
    parser = argparse.ArgumentParser(description='Run the ForceCRM analysis as cached stages: ' + ' → '.join(ORDER))
    parser.add_argument('--portfolio', type=Path)
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('outputs/cli'))
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--model', default=DEFAULT_LLM)
    parser.add_argument('--bootstrap', type=int, default=20)
    parser.add_argument('--from-stage', choices=ORDER, help='recompute this stage and everything after it, ignoring cached artifacts')
    parser.add_argument('--until-stage', choices=ORDER, help='stop after this stage (and what it needs); prints the stage log, writes no exports')
    parser.add_argument('--filters', type=json.loads, metavar='JSON',
                        help='exclusions, e.g. \'{"mode": "post", "activities": ["5630"], "prefectures": ["ΚΥΚΛΑΔΩΝ"]}\'; '
                             'mode "pre" filters the raw data, "post" only the suggestions')
    parser.add_argument('--no-cache', action='store_true', help='do not read or write stage artifacts on disk')
    args = parser.parse_args()
    if not args.demo and not args.portfolio:
        parser.error('--portfolio is required unless --demo is selected')
    if args.bootstrap < 2:
        parser.error('--bootstrap must be at least 2')
    c, a, _, _ = load_universe(args.data)
    portfolio = demo_portfolio(c, a) if args.demo else read_portfolio(args.portfolio, c)
    force = downstream(args.from_stage) if args.from_stage else ()
    options = dict(demo=args.demo, model=args.model, bootstrap=args.bootstrap, progress=lambda m: print(m, flush=True),
                   cache=not args.no_cache, force=force)
    if args.until_stage:
        filters, portfolio, _ = prepare_filters(args.data, portfolio, args.filters)
        runner = make_runner(args.data, portfolio, filters=filters, **options)
        with threadpool_limits(limits=2, user_api='blas'):
            runner.get(args.until_stage)
        print(json.dumps(runner.summary(), indent=1))
        return
    result = run_analysis(args.data, portfolio, filters=args.filters, **options)
    export_result(result, args.output)
    for stage in result['report']['stages']:
        print(f"  {stage['stage']:<10} {stage['status']:<9} {stage['seconds']:>8.2f} s  {stage['key']}")
    print(f'Saved {len(result["profiles"])} profiles and {len(result["lookalikes"])} candidates to {args.output}')


if __name__ == '__main__':
    main()
