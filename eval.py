import argparse
import nltk

def main():
    # Command line args
    args = parse_args()
    # Label name
    lname = "i"
    # True positive, false positive, false negative
    tp, fp, fn = 0, 0, 0
    true_pos,false_pos,false_neg= [],[],[]

    # Open hyp and ref tsv files
    with open(args.hyp) as hyp_tsv, open(args.ref) as ref_tsv:
        # Process line by line in both files
        for i, (hline, rline) in enumerate(zip(hyp_tsv, ref_tsv)):
            # Get the hyp and ref info
            hyp_info = hline.strip()
            ref_info = rline.strip()
            # Ignore empty lines
            if not hyp_info or not ref_info: continue
            # Split on tab
            hyp_info = hyp_info.split("\t")
            ref_info = ref_info.split("\t")
            # Make sure this is the same token in both files
            assert hyp_info[0] == ref_info[0]
            # Get the hyp label and ref label
            hyp_label = hyp_info[1]
            ref_label = ref_info[1]
            # True Positive
            if hyp_label == ref_label == lname:
                tp += 1
                true_pos.append((ref_info[0],i))

            # Non-matching labels
            if hyp_label != ref_label:
                # False positive
                if hyp_label == lname:
                    fp += 1
                    false_pos.append((ref_info[0],i))
                # False negative
                if ref_label == lname:
                    fn += 1
                    false_neg.append((hyp_info[0],i))

    #print("Tokens detected as incorrect that were in fact correct:\n",false_pos)
    #print("Tokens detected as correct that were in fact incorrect:\n", false_neg)
    # Calculate Precision, Recall and F_beta
    p, r, f = compute_fscore(tp, fp, fn, args.beta)
    # Print the overall results.
    print("")
    print('{:=^46}'.format(" Token-Based Detection "))
    print("\t".join(["TP", "FP", "FN", "Prec", "Rec", "F"+str(args.beta)]))
    print("\t".join(map(str, [tp, fp, fn, p, r, f])))
    print('{:=^46}'.format(""))
    print("")

    indexes_false_pos=[token_index[1] for token_index in false_pos]
    indexes_false_neg=[token_index[1] for token_index in false_neg]

    replaced_content=""
    if args.ann:
        with open(args.ann) as f:
            l=f.readlines()
            for i, line in enumerate(l):
                if i in indexes_false_pos:
                    new_line=line[:-1]+" FALSE POSITIVE"
                    replaced_content=replaced_content+new_line+"\n"
                elif i in indexes_false_neg:
                    new_line=line[:-1]+ " FALSE NEGATIVE"
                    replaced_content=replaced_content+new_line+"\n"
                else:
                    new_line=line[:-1]
                    replaced_content=replaced_content+new_line+"\n"

        with open(args.ann, 'w') as f:
                f.write(replaced_content)

# Parse command line args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate F-score for token-based error detection.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [options] -hyp HYP -ref REF -ann ANN")
    parser.add_argument(
        "-hyp",
        help="A hypothesis tsv file.",
        required=True)
    parser.add_argument(
        "-ref",
        help="A reference tsv file.",
        required=True)
    parser.add_argument(
        "-b",
        "--beta",
        help="Value of beta in F-score. (default: 0.5)",
        default=0.5,
        type=float)
    parser.add_argument(
        "--ann",
        "--annotation",
        help="The file where the errors will be annotated",
        default=None
    )
    args = parser.parse_args()
    return args

# Input 1-3: True positives, false positives, false negatives
# Input 4: Value of beta in F-score.
# Output 1-3: Precision, Recall and F-score rounded to 4dp.
def compute_fscore(tp, fp, fn, beta):
    p = float(tp)/(tp+fp) if fp else 1.0
    r = float(tp)/(tp+fn) if fn else 1.0
    f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

if __name__ == "__main__":
    main()
