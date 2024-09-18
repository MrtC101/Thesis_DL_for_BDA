function get() {
    # Recibe URL y usuario como argumentos
    local user_url=$1

    if [[ -z $user_url ]]; then
        echo "Usage: re user@url"
        return 1
    fi

    # Recibe archivos desde el servidor remoto
    rsync -avz \
    --exclude-from=/home/mrtc101/Desktop/tesina/repo/to_mendieta/submit/exclude.txt \
    "${user_url}:scratch/" .
}

function send() {
    # Recibe URL y usuario como argumentos
    local user_url=$1

    if [[ -z $user_url ]]; then
        echo "Usage: send user@url"
        return 1
    fi

    # Elimina caches y envía archivos
    find . -type d -name '__pycache__' -exec rm -r {} +
    rsync -avz \
    --exclude-from=/home/mrtc101/Desktop/tesina/repo/to_mendieta/submit/exclude.txt \
    . "${user_url}:scratch/"
}

alias send-toko='send mcogo@toko.uncu.edu.ar'
alias get-toko='get mcogo@toko.uncu.edu.ar'
alias send-mendieta='send mcogo@mendieta.ccad.unc.edu.ar'
alias get-mendieta='get mcogo@mendieta.ccad.unc.edu.ar'