mkdir -p ../models/_pretrained_weights/flowformer_weights

# kitti
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10RKHdF1mlbf2JV7_DUz14wV5VbG2W1rY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10RKHdF1mlbf2JV7_DUz14wV5VbG2W1rY" -O ../models/_pretrained_weights/flowformer_weights/kitti.pth && rm -rf /tmp/cookies.txt
# sintel
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Vtwk71rFhFQLY39e-guPPuG5OLevonhR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Vtwk71rFhFQLY39e-guPPuG5OLevonhR" -O ../models/_pretrained_weights/flowformer_weights/sintel.pth && rm -rf /tmp/cookies.txt
# things
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iROet3XsII7GXMkTgjkU4kt_Kg-jQDVf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iROet3XsII7GXMkTgjkU4kt_Kg-jQDVf" -O ../models/_pretrained_weights/flowformer_weights/things.pth && rm -rf /tmp/cookies.txt
# things-kitti
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xBueXJ4RzDi73Bn2tVO9qZR93flR5WwX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xBueXJ4RzDi73Bn2tVO9qZR93flR5WwX" -O ../models/_pretrained_weights/flowformer_weights/things-kitti.pth && rm -rf /tmp/cookies.txt
# chairs
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v9SHsLw4IYVGARRerOJB-RsLnKBdc1We' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v9SHsLw4IYVGARRerOJB-RsLnKBdc1We" -O ../models/_pretrained_weights/flowformer_weights/chairs.pth && rm -rf /tmp/cookies.txt
