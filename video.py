from moviepy.editor import ImageSequenceClip
import argparse
from moviepy.editor import VideoFileClip
import os
import cv2

def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


def extract_frames(movie, imgdir):
    
    clip = VideoFileClip(movie)
    frano = 0
    for frames in clip.iter_frames():
     
        imgpath = os.path.join(imgdir, '{}.png'.format(str(frano).zfill(5)))
        cv2.imwrite(imgpath, cv2.cvtColor(frames, cv2.COLOR_RGB2BGR))
        print("Saved to: ", imgpath)
        frano = frano +1

if __name__ == '__main__':
    
    movie = 'C:\GIT\CarND-Advanced-Lane-Lines\harder_challenge_video.mp4'
    imgdir = 'C:/GIT/SDC_projects_P12_advanced_deep_learning/video_images_hc/'
    extract_frames(movie, imgdir)


#    #my_clip. 
#    video_output1 = 'project_video_output.mp4'
#    video_input1 = VideoFileClip('project_video.mp4')#.subclip(22,26)
#    processed_video = video_input1.fl_image(process_image)
#    #processed_video.write_videofile(video_output1, audio=False)
#    processed_video.write_gif('test.gif', fps=12)