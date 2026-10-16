"""Create a downloadable archive of public evaluation outputs and logs."""
import tarfile,os
with tarfile.open('/content/035r.tar.gz','w:gz') as tar:
 if os.path.isdir('/content/035r'):tar.add('/content/035r',arcname='results')
 if os.path.isfile('/content/infer.log'):tar.add('/content/infer.log',arcname='results/infer.log')
print('/content/035r.tar.gz ready')
