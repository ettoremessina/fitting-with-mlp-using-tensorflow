while read -rn1;do printf '%s' "$REPLY"; sleep 0.1; done<<<"$1"
printf '\n'
