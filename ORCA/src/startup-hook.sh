CURPATH=`pwd`

# OTDD 
#pip install gdown
cd otdd
pip install -r requirements.txt
pip install .

# for FSD50K
# pip install librosa
# apt-get install -y libsndfile1

cd $CURPATH


