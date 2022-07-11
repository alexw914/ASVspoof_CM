wget https://www.asvspoof.org/resources/LA-keys-stage-1.tar.gz
wget https://www.asvspoof.org/resources/DF-keys-stage-1.tar.gz

rm -R keys
tar -zxvf LA-keys-stage-1.tar.gz
mv keys LA 
tar -zxvf DF-keys-stage-1.tar.gz 
mv keys DF

mkdir keys
cd keys && mkdir 2019LA LA DF && cd ../
ln -s /home/alex/Corpora/ASVspoof2019/LA/ASVspoof2019_LA_asv_scores ./keys/2019LA/ASV
mv LA/* ./keys/LA && mv DF/* ./keys/DF

rm -R LA DF