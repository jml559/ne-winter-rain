#!/bin/sh

MDL=${1:-"/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public"}
OUT=${2:-"${MDL}/working"}
  DSP=${3:-"NO"}
cd $OUT
echo "### ${0##*/} ###"

syr=1983
smo=09
data="RP${syr}${smo}.gbl"

USER=anonymous
PASS=name@server.com

ftp -n ftp.arl.noaa.gov <<EOD
user $USER ${PASS}
cd /pub/archives/reanalysis
binary
prompt
get ${data}
bye
EOD
