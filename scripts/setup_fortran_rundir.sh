#!/usr/bin/env bash
# Set up and launch one OpenIFS offline-driver (osmMASTER) run directory.
#
# Usage: setup_fortran_rundir.sh RUNDIR FORCING_PATH NSTOP NFRPOS NFRRES YEAR NDFORC [THREADS]
#
# The namelist replicates the archived fortran_reference_1y_2018 experiment;
# only the step counts, date, and forcing record count vary.
set -euo pipefail

RUNDIR=$1
FORCING=$2
NSTOP=$3
NFRPOS=$4
NFRRES=$5
YEAR=$6
NDFORC=$7
THREADS=${8:-128}

INIT=${SURF_INIT:-/data/yangjinhui/surf_pytorch/surf_paper/experiments/initial_states/native_20d_terminal_complete.nc}
EXE=/home/qixiang/yangjinhui/openifs/build_offline/bin/osmMASTER

mkdir -p "$RUNDIR"
cd "$RUNDIR"
if [ "$(readlink -f "$FORCING")" != "$(readlink -f forcing 2>/dev/null || true)" ]; then
    ln -sf "$FORCING" forcing
fi
ln -sf "$INIT" restartin.nc
ln -sf restartin.nc soilinit
ln -sf restartin.nc surfclim

cat > input <<EOF
&NAMCT01S
 NSTART=0, NSTOP=$NSTOP, NFRPOS=$NFRPOS, NFRRES=$NFRRES, LNF=.TRUE., NCYCLE=1, CNMEXP="CMFD_10Y" /
&NAMDYN1S
 TSTEP=1800, NACCTYPE=2, LSWINT=.FALSE., LPREINT=.FALSE., LFLXINT=.FALSE., TCOUPFREQ=86400. /
&NAMDIM
 NLAT=1, NLON=97709, NDFORC=$NDFORC, NCSS=4, NCSNEC=1 /
&NAMRIP
 NINDAT=${YEAR}0101, NSSSSS=0 /
&NAM1S
 CFFORC='netcdf', CFOUT='netcdf', CFSURF='netcdf', CFINIT='netcdf', CMODID='HTESSEL',
 LACCUMW=.TRUE., LRESET=.TRUE., NACCUR=1, NDIMCDF=2, NDLEVEL=0, NCDFTYPE=4,
 LWRGG=.TRUE., LWRCLM=.FALSE., LWRGGD=.FALSE., LWREFL=.FALSE., LWRWAT=.FALSE., LWRD2M=.FALSE.,
 LWRSUS=.FALSE., LWREVA=.FALSE., LWRCLD=.FALSE., LWRCO2=.FALSE., LWRBIO=.FALSE., LWRVEG=.FALSE.,
 LWRVTY=.FALSE., LWRTIL=.FALSE., LWRLKE=.FALSE., LSEMISS=.FALSE., LDBGS1=.FALSE., IDBGS1=1, LNCSNC=.FALSE. /
&NAMFORC
 ZPHISTA=10.0, ZUV=10.0, ZDTFORC=10800, INSTFC=0, NDIMFORC=2, LOADIAB=.FALSE.,
 CFORCV='forcing', CFORCU='forcing', CFORCT='forcing', CFORCQ='forcing', CFORCP='forcing',
 CFORCRAIN='forcing', CFORCSNOW='forcing', CFORCSW='forcing', CFORCLW='forcing' /
&NAMPHY
 LEVGEN=.TRUE., LESSRO=.TRUE., LEFLAKE=.FALSE., LESN09=.TRUE., LELAIV=.FALSE., LECTESSEL=.FALSE.,
 LEAGS=.FALSE., LEFARQUHAR=.TRUE., LEOPTSURF=.FALSE., LEAIRCO2COUP=.FALSE., LEC4MAP=.TRUE., RLAIINT=0,
 LECLIM10D=.FALSE., LEINTWIND=.TRUE., LESNML=.FALSE., LECMF1WAY=.FALSE. /
&NAMOPTSURF /
&NAMPHYOFF
 LEWBCHECK=.FALSE., LEWBCHECKAbort=.FALSE., LESNCHECK=.FALSE., LESNCHECKAbort=.FALSE., LEWBSOILFIX=.TRUE.,
 LESKTI5=.FALSE., LESKTI8=.FALSE., LEWARMSTART=.FALSE., LECOLDSTART=.FALSE., LESOILCOND=.TRUE., LESNWBCON=.FALSE., LEROLAKE=.TRUE. /
&NAMPARSNOW /
&NAMPARSOIL /
&NAMPARVEG /
&NAMPARAGS /
&NAMPARFLAKE /
&NAMPAREXC /
&NAMPARURB /
EOF

echo "run directory ready: $RUNDIR (NSTOP=$NSTOP NDFORC=$NDFORC threads=$THREADS)"
