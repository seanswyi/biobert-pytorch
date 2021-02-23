import argparse
import ast
import os
import subprocess
import time


def main(args):
    start_time = time.time()

    all_results = {'NER': {'BC5CDR-chem': {},
                           'BC5CDR-disease': {}},
                   'RE': {'DDI': {},
                          'ChemProt': {}},
                   'NLI': {'MedNLI': {}}}

    for task in ['NER', 'RE', 'NLI']:
        if task == 'NER':
            ner_dir = os.path.join(args.root_dir, args.ner_dir)

            for entity in ['BC5CDR-chem', 'BC5CDR-disease']:
                command = ['bash']
                command.extend([os.path.join(ner_dir, 'run_ner.sh'), entity, args.model_name_or_path])

                ner_results = subprocess.check_output(command)
                ner_output = ast.literal_eval(ner_results.decode('utf-8'))

                all_results['NER'][entity] = ner_output
        elif task == 'RE':
            re_dir = os.path.join(args.root_dir, args.re_dir)

            for entity in ['DDI', 'ChemProt']:
                command = ['bash']
                command.extend([os.path.join(re_dir, 'run_re.sh'), entity, args.model_name_or_path])

                re_results = subprocess.check_output(command)
                re_output = ast.literal_eval(re_results.decode('utf-8'))

                all_results['RE'][entity] = re_output
        elif task == 'NLI':
            nli_dir = os.path.join(args.root_dir, args.nli_dir)

            for entity in ['MedNLI']:
                command = ['bash']
                command.extend([os.path.join(nli_dir, 'run_nli.sh'), entity, args.model_name_or_path])

                nli_results = subprocess.check_output(command)
                nli_output = ast.literal_eval(nli_results.decode('utf-8'))

                all_results['NLI'][entity] = nli_output
        else:
            raise NotImplementedError

    end_time = time.time()
    total_time = end_time - start_time
    total_time = time.strftime('%H:%M:%S', time.gmtime(total_time))

    model_name = args.model_name_or_path.split('/')[-1]
    output = [model_name]
    all_f1 = []
    for task in all_results:
        for entity in all_results[task]:
            try:
                output.append(f"{all_results[task][entity]['eval_f1']:.6f}")
                all_f1.append(all_results[task][entity]['eval_f1'])
            except KeyError:
                output.append(f"{all_results[task][entity]['f1']:.6f}")
                all_f1.append(all_results[task][entity]['f1'])

            output.append(str(float(all_results[task][entity]['training_time'].replace(':', ''))))

    average_f1 = sum(all_f1) / len(all_f1)
    all_training_times = list(map(float, [all_results[task][entity]['training_time'].replace(':', '') for task in all_results for entity in all_results[task]]))
    total_training_time = sum(all_training_times)

    output.append(f"{average_f1:.6f}")
    output.append(str(total_training_time))

    output = ' '.join(output)

    print(f'output = {output}')
    print(f"Total process took {total_time}.")

    save_filename = os.path.join(args.root_dir, args.save_file)
    with open(file=save_filename, mode='a') as f:
        f.write(output + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/home/seokwon/github/biobert-pytorch', type=str)
    parser.add_argument('--ner_dir', default='named-entity-recognition', type=str)
    parser.add_argument('--re_dir', default='relation-extraction', type=str)
    parser.add_argument('--nli_dir', default='natural-language-inference', type=str)
    parser.add_argument('--log_dir', default='results', type=str)
    parser.add_argument('--task', default='NER', type=str)
    parser.add_argument('--ner_entity', default='BC5CDR-chem', type=str, choices=['BC5CDR-chem', 'BC5CDR-disease'])
    parser.add_argument('--data_dir', default='datasets', type=str)
    parser.add_argument('--model_name_or_path', default='bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12', type=str)
    parser.add_argument('--output_dir', default='/hdd1/seokwon/bluebert_pytorch_outputs/ner', type=str)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--per_device_train_batch_size', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_file', default='downstream_results.txt', type=str)

    args = parser.parse_args()

    main(args)
