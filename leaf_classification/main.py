from train import classify
import argparse
import datetime

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_name",type=str,default='4.JPG')
    parser.add_argument("--save_result",type=bool,default=True)
    args=parser.parse_args()
    image_name=args.image_name
    label,out=classify(image_name)
    if args.save_result:
        time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file=open("predict_result.txt","a")
        file.write(image_name+" ")
        for n in range(5):
            print label[n],out[n]
            file.write(str(label[n])+" "+str(out[n])+" ")
        file.write(time+"\n")
        file.close()

