#! /bin/bash

####### abandoned in favor of parse_results.sh #######

# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <dataset> <results-filepath> <output-filepath>"
#     exit 1
# fi

for gan in cramergan mmdgan sngan progan real
do
    if [ "${gan}" = "real" ]; then
        target="${1}_real_data"
        extension="jpg"
    else
        target="${gan}_generated_data"
        extension="png"
    fi
    tp=$(grep "${gan}_[0-9]*\.${extension}" $2 | grep -i "${target}" | wc -l)
    tn=$(grep -v "${gan}_[0-9]*\.${extension}" $2 | grep -iv "${target}" | grep -i "^classifying" | wc -l)
    fp=$(grep "${gan}_[0-9]*\.${extension}" $2 | grep -iv "${target}" | grep -i "^classifying" | wc -l)
    fn=$(grep -v "${gan}_[0-9]*\.${extension}" $2 | grep -i "${target}" | wc -l)
    pos=$(grep "${gan}_[0-9]*\.${extension}" $2 | wc -l)
    neg=$(grep -v "${gan}_[0-9]*\.${extension}" $2 | grep -i "^classifying" | wc -l)
    cat << EOF
${gan}:
  tpr: ${tp}/${pos}
  tnr: ${tn}/${neg}
  fpr: ${fp}/${pos}
  fnr: ${fn}/${neg} # some of these calcs are wrong
EOF
done