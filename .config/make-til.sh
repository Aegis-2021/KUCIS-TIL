#!/bin/sh

NAME=$1
TODAY=`date +"%y-%m-%d"`
DIR_PATH="./TIL/${NAME}"
NEW_FILE="${NAME}(${TODAY}).md"

if [ ! -d "./TIL" ]; then
    mkdir ./TIL
fi

if [ ! -d "${DIR_PATH}" ]; then
    mkdir "${DIR_PATH}"
    mkdir "${DIR_PATH}/images"
fi

if [ -e "${DIR_PATH}/${NEW_FILE}" ]; then
    echo "Aegis > ${NEW_FILE}이 이미 존재합니다."
    exit 0
fi

cp template/TIL_TEMPLATE.md "${DIR_PATH}/${NEW_FILE}"
