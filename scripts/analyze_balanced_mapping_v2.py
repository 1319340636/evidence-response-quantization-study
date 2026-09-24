"""Source-verified mapping sensitivity using the existing 12-endpoint estimator."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

from qer_fv.vitaminc_balanced_mapping_gate import build_balanced_mapping_gate
from qer_fv.vitaminc_balanced_mapping_protocol import load_balanced_mapping_units, EXPECTED_MAPPINGS
from qer_fv.prompts import load_prompt_contract, render_prompt
from qer_fv.vitaminc_mapping_analysis import mapping_interaction_margin
from qer_fv.three_model_scale_heterogeneity import analyze_three_model_scale_heterogeneity
from qer_fv.major_revision_scale import classify_scale_robust_gate
from qer_fv.multiplicity import holm_adjust
from qer_fv.three_model_scale_sensitivity import analyze_three_model_scale_sensitivity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reestimate-scale', action='store_true')
    a = p.parse_args()
    root = a.root
    run = root/'runs/vitaminc_hf_balanced_mapping_v2'
    gate = build_balanced_mapping_gate(root, run/'formal', population='formal')
    saved = json.loads((run/'formal_gate_v2.json').read_text())
    assert gate == saved, 'gate changed'
    units, _ = load_balanced_mapping_units(root, population='formal')
    expected = {(u.case_id, u.cell_index): u for u in units}
    contract = load_prompt_contract(root/'configs/prompt_contract_v2.json')
    results = {}
    accuracy = {}
    models = {'qwen':'qwen35_9b','ministral':'ministral3_8b','olmo':'olmo3_7b'}
    routes = {'fp16':'FP16','gptq':'GPTQ_INT4','awq':'AWQ_INT4'}
    for mapping, labels in EXPECTED_MAPPINGS.items():
        values = {}
        pages = {}
        accuracy[mapping] = {}
        for model, key in models.items():
            values[model] = {}
            for route, precision in routes.items():
                path = run/'formal'/mapping/(key+'_'+precision)/'export/records.jsonl'
                cells = defaultdict(dict)
                correct = 0
                seen = set()
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    m = row['metadata']
                    pair = (m['case_id'], m['cell_index'])
                    assert pair not in seen
                    seen.add(pair)
                    u = expected[pair]
                    assert m['mapping_variant'] == mapping and m['page'] == u.page
                    assert m['negative_label'] == u.prompt_input.metadata['negative_label']
                    rendered = render_prompt(contract, u.prompt_input, mapping_variant=mapping)
                    h, q = row['hard_payload'], row['probability_payload']
                    assert h['prompt_sha256'] == q['prompt_sha256'] == rendered.prompt_sha256
                    assert h['choice_logprobs'] == q['choice_logprobs']
                    assert h['scored_label'] == labels[h['scored_choice']]
                    assert m['gold_label'] == u.prompt_input.metadata['gold_label']
                    correct += h['scored_label'] == m['gold_label']
                    cells[u.case_id][u.cell_index] = mapping_interaction_margin(q['choice_logprobs'], negative_label=m['negative_label'], choice_to_label=labels)
                    pages[u.case_id] = u.page
                assert seen == set(expected)
                assert all(set(v)=={0,1,2,3} for v in cells.values())
                values[model][route] = {c:(v[0]-v[1]-v[2]+v[3])/2 for c,v in cells.items()}
                accuracy[mapping][model+'_'+route] = correct/4800
        assert len(pages)==1200 and len(set(pages.values()))==1078
        print('validated '+mapping, flush=True)
        estimator = analyze_three_model_scale_sensitivity if a.reestimate_scale else analyze_three_model_scale_heterogeneity
        results[mapping] = estimator(model_values=values, pages=pages)
    if a.reestimate_scale:
        for weighting in ('page_balanced', 'quartet_weighted'):
            raw = {}
            for m in ('cycle_1', 'cycle_2'):
                for pair, cs in results[m]['weighting_sensitivities'][weighting]['interactions'].items():
                    for contrast, item in cs.items():
                        for ep, v in item['endpoints'].items():
                            raw[f'{m}.{pair}.{contrast}.{ep}'] = v['raw_pvalue']
            assert len(raw) == 24
            adjusted = holm_adjust(raw)
            for m in ('cycle_1', 'cycle_2'):
                family = results[m]['weighting_sensitivities'][weighting]
                family['holm_tests'] = 24
                for pair, cs in family['interactions'].items():
                    for contrast, item in cs.items():
                        for ep, v in item['endpoints'].items():
                            v['holm_adjusted_pvalue'] = adjusted[f'{m}.{pair}.{contrast}.{ep}']
                        s = item['endpoints']['reference_standardized_route_effect']
                        d = item['endpoints']['directional_common_language_effect']
                        item['claim_gate'] = classify_scale_robust_gate(standardized_estimate=s['estimate'], standardized_adjusted_pvalue=s['holm_adjusted_pvalue'], directional_estimate=d['estimate'], directional_adjusted_pvalue=d['holm_adjusted_pvalue'])
        out = root/'reports/vitaminc_hf_balanced_mapping_v2'
        out.mkdir(parents=True, exist_ok=True)
        dest = out/'scale_sensitivity.json'
        record = {'formal_gate_sha256': saved['gate_sha256'], 'analysis': results,
                  'analysis_role': 'post_review_sensitivity_not_frozen_primary',
                  'multiplicity': '24 alternative-mapping endpoints per weighting; original 12 separately; weightings are sensitivity analyses',
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        dest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
        dest.with_suffix('.json.sha256').write_text(hashlib.sha256(dest.read_bytes()).hexdigest()+'\n')
        print('scale sensitivity complete', flush=True)
        return
    raw = {m+'.'+x['name']:x['raw_pvalue'] for m in ('cycle_1','cycle_2') for x in results[m]['holm_family']}
    assert len(raw)==24
    adj = holm_adjust(raw)
    for m in ('cycle_1','cycle_2'):
        for pair, contrasts in results[m]['interactions'].items():
            for contrast, item in contrasts.items():
                for ep, v in item['endpoints'].items():
                    v['holm_adjusted_pvalue'] = adj[m+'.'+pair+'.'+contrast+'.'+ep]
                s = item['endpoints']['reference_standardized_route_effect']
                d = item['endpoints']['directional_common_language_effect']
                item['claim_gate'] = classify_scale_robust_gate(standardized_estimate=s['estimate'], standardized_adjusted_pvalue=s['holm_adjusted_pvalue'], directional_estimate=d['estimate'], directional_adjusted_pvalue=d['holm_adjusted_pvalue'])
        for x in results[m]['holm_family']:
            x['holm_adjusted_pvalue'] = adj[m+'.'+x['name']]
        results[m]['bootstrap_contract']['holm_tests'] = 24
    record = {'formal_gate_sha256': saved['gate_sha256'], 'analysis':results, 'descriptive_accuracy':accuracy, 'multiplicity':'24 alternative-mapping endpoints jointly Holm adjusted; original 12 separately', 'scale_policy':'existing fixed-reference-SD estimator; intervals descriptive; denominator-reestimated sensitivity not computed here'}
    out = root/'reports/vitaminc_hf_balanced_mapping_v2'
    out.mkdir(parents=True,exist_ok=True)
    dest = out/'results.json'
    dest.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (out/'results.json.sha256').write_text(hashlib.sha256(dest.read_bytes()).hexdigest()+'\n')
    for m,r in results.items():
        for pair,cs in r['interactions'].items():
            for contrast,item in cs.items():
                print(m,pair,contrast,item['claim_gate'],[(k,round(v['estimate'],4),v['holm_adjusted_pvalue']) for k,v in item['endpoints'].items()],flush=True)


if __name__ == '__main__':
    main()
