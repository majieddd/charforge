#!/bin/zsh
# A parallel improvement lane: its own git worktree and branch, the shared heavy folders linked, its test characters cloned.
#   zsh setup_lane.sh <lane> <character> [<character> ...]
# -> <the folder holding charforge/>/cf_lanes/<lane>  (branch lane/<lane>, from main; CHARFORGE_PARENT overrides the folder)
#    vendor/ models/ -> the main checkout's (read-only use); ../.venv -> the project venv
#    work/.gpu.lock and work/motion_library.npz -> the main checkout's (one GPU job at a time across every lane)
#    work/<c> out/<c> -> APFS copy-on-write clones of the main checkout's (no extra space until changed)
set -e
TL=${CHARFORGE_PARENT:-$(cd "$(dirname "$0")/../../.." && pwd)}
CF=$TL/charforge
L=$1; shift
WT=$TL/cf_lanes/$L
mkdir -p $TL/cf_lanes
[[ -e $TL/cf_lanes/.venv ]] || ln -s $TL/.venv $TL/cf_lanes/.venv
if [[ ! -d $WT ]]; then
  git -C $CF worktree add -q $WT -b lane/$L main
fi
for d in vendor models; do [[ -e $WT/$d ]] || ln -s $CF/$d $WT/$d; done
mkdir -p $WT/work $WT/out
for f in .gpu.lock motion_library.npz; do [[ -e $WT/work/$f ]] || ln -s $CF/work/$f $WT/work/$f; done
for c in $@; do
  [[ -d $WT/work/$c ]] || cp -cR $CF/work/$c $WT/work/$c
  [[ -d $WT/out/$c || ! -d $CF/out/$c ]] || cp -cR $CF/out/$c $WT/out/$c
done
echo "lane $L ready at $WT with: $@"
