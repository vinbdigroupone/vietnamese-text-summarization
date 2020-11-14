import rouge
from Dataset import VietDataset, make_dataloader
from tqdm import tqdm

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate_vi(path, aggregator, model):
    ''' Evaluate model using rouge score

    Parameters:
        path: (String) path to folder that contains test data
        aggregator: (String) 'Avg' or 'Best'
        model: Summarize-model. model.summarize() should receive a String input and
            return a String output
    '''

    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'
    evaluator = rouge.Rouge(metrics=['rouge-n'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
    dataset = VietDataset(path)
    dataloader = make_dataloader(dataset)

    # Loop over all input_text in dataloader
    all_outputs, all_targets = list(), list()
    for i, (text, target) in enumerate(tqdm(dataloader)):
        if i > 5:
            break
        target = target[0] 
        text = text[0] 

        output = model.summarize(text)

        all_outputs.append(output)
        all_targets.append(target)

    scores = evaluator.get_scores(all_outputs, all_targets)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f'], metric))
    print()