if [[ -d "replay-memory" ]]; then
    rm -r replay-memory
    echo "replay-memory clean"
fi
if [[ -d "train/qnet" ]]; then
    rm -r train/qnet
    echo "train/qnet clean"
fi
if [[ -n $(ls ckpts/qnet) ]]; then
    rm ckpts/qnet/*
    echo "ckpts/qnet clean"
fi
rm -r algorithm/ocr
