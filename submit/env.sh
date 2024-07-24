function re() {
    # recive files from toko
    rsync -avz \
    --exclude 'cpu_exp/data' \
    --exclude 'cpu_exp/out/*/*/checkpoints' \
    --exclude 'cpu_exp/out/*/checkpoints' \
    --exclude '*/__pycache__/*' \
    mcogo@toko.uncu.edu.ar:scratch/ .
}

function send() {
    # deletes caches and send files
    find . -type d -name '__pycache__' -exec rm -r {} +
    rsync -avz \
    --exclude .vscode \
    --exclude .ignore \
    --exclude .git \
    --exclude data \
    --exclude recive.sh \
    --exclude send.sh \
    --exclude out \
    --exclude submit/env.sh \
    --exclude notebooks \
    --exclude todo.txt \
    --exclude README.md \
    --exclude LICENSE \
    . mcogo@toko.uncu.edu.ar:scratch/
}
alias send=send
alias re=re