#!/bin/bash
# download weights for flownetc and store under pretrained 
FILENAME="FlowNet2-C_checkpoint.pth.tar"
FILEID="1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
mv  $FILENAME ../models/_pretrained_weights