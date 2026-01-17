#include "MMMSSpeed.h"
#include <filesystem>
#include <inttypes.h>
#include "logger.h"
#include "fmt/format.h"

static int counter = 0;  //to count video frames
static int rawCounter = 0; //to create raw video frame file
static int counter2 = 0; //to count audio frames

MMMSSpeed::MMMSSpeed():playback_type(""),speed_value(0.0),filter_graph(nullptr),buffersrc_ctx(nullptr),buffersink_ctx(nullptr),
                       adjustVideoPlayback(false), adjustAudioPlayback(false), noAudioFlag(false)
{

    avFramesBufferAudio = {};
    avFramesBufferFilteredAudio = {};
}

void MMMSSpeed::InitMetaData(MMMetaData mmxMD, MMStatistics& stats)
{
  MMMMSConversion::InitMetaData(mmxMD, stats);
  playback_type = mmxMD.playbackType;
  speed_value = mmxMD.speed;
  adjustAudioPlayback = (bool)mmxMD.adjustAudioPlayback;
  adjustVideoPlayback = (bool)mmxMD.adjustVideoPlayback;
  noAudioFlag = (bool) mmxMD.noAudioFlag;
}

int MMMSSpeed::prepare_audio_decoder()
{
    if (!noAudioFlag)
    {
        return MMMMSConversion::prepare_audio_decoder();
    }
    return 0;
}

int MMMSSpeed::prepare_audio_encoder(int numChannels, uint64_t bitRate, AVCodecID audio_codecID)
{
    if (!noAudioFlag)
    {
        return MMMMSConversion::prepare_audio_encoder(numChannels, bitRate, audio_codecID);
    }
    return 0;
}

int MMMSSpeed::encode_audio(AVFrame *input_frame)
{
    if (!noAudioFlag)
    {
        return MMMMSConversion::encode_audio(input_frame);
    }
    return 0;
}

int MMMSSpeed::transcode_audio(AVFrame *iframe, AVPacket *input_packet, int is_raw_needed)
{
    if (!noAudioFlag)
    {
        if(!adjustAudioPlayback)
            return MMMMSConversion::transcode_audio(iframe, input_packet, is_raw_needed);
        else
        {
            int response = avcodec_send_packet(input_codec_context_audio, input_packet);
            if (response < 0) {
                Logger::Error("Error while sending packet to decoder:",__FILE__,__LINE__); //av_err2str(response));
                return response;
            }

            while (response >= 0) {
                response = avcodec_receive_frame(input_codec_context_audio, iframe);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                    break;
                }
                else if (response < 0) {
                    Logger::Error("Error while receiving packet to decoder:", __FILE__, __LINE__); // av_err2str(response));
                    return response;
                }

               storeAVFrameAudio(iframe);
            }
            return 0;
        }
    }
    return 0;
}

int MMMSSpeed::ffmpeg_filter_graph_intialization(double speedx) {
    char args[512];
    int ret = 0;

    filter_graph = avfilter_graph_alloc();
    if (!filter_graph) {
        Logger::Error("Could not allocate Filter graph,", __FILE__, __LINE__);
        return -1;
    }

    snprintf(args, sizeof(args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d",
        input_codec_context_video->width, input_codec_context_video->height, input_codec_context_video->pix_fmt,
        input_codec_context_video->time_base.num, input_codec_context_video->time_base.den);

    if (avfilter_graph_create_filter(&buffersrc_ctx, avfilter_get_by_name("buffer"), "src",
                                     args, nullptr, filter_graph) < 0) {
        Logger::Error("Cannot create buffer source.", __FILE__, __LINE__);
        return -1;
    }

    if (avfilter_graph_create_filter(&buffersink_ctx, avfilter_get_by_name("buffersink"), "sink",
                                     nullptr, nullptr, filter_graph) < 0) {
        Logger::Error("Cannot create buffer sink", __FILE__, __LINE__);
        return -1;
    }

    AVFilterInOut* outputs = avfilter_inout_alloc();
    AVFilterInOut* inputs  = avfilter_inout_alloc();
    outputs->name       = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx;
    outputs->pad_idx    = 0;
    outputs->next       = nullptr;

    inputs->name       = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx;
    inputs->pad_idx    = 0;
    inputs->next       = nullptr;

    if(playback_type == "fast") {
      speedx = 1/speedx;
    }
    std::string pts_value = std::to_string(speedx);
    std::string pts_config = "setpts=" + pts_value + "*PTS";

    Logger::Info(fmt::format("PTS CONFIGURATION: {}", pts_config), __FILE__, __LINE__);

    if (avfilter_graph_parse_ptr(filter_graph, pts_config.c_str(), &inputs, &outputs, nullptr) < 0) {
        Logger::Error("Failed to parse filter graph",__FILE__,__LINE__);
        return -1;
    }

    if (avfilter_graph_config(filter_graph, nullptr) < 0) {
        Logger::Error("Failed to configure filter graph",__FILE__,__LINE__);
        return -1;
    }

    Logger::Info("Filter Graph intialization is successful",__FILE__,__LINE__);

    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);

    return 0;
}

// Function to create a new AVFrame with the desired configuration
AVFrame* MMMSSpeed::create_new_frame(int nb_samples, int channels, AVSampleFormat sample_fmt, int sample_rate) {
    AVFrame* frame = av_frame_alloc();
    frame->nb_samples = nb_samples;
    frame->channel_layout = av_get_default_channel_layout(channels);
    frame->format = sample_fmt;
    frame->sample_rate = sample_rate;
    av_frame_get_buffer(frame, 0);
    return frame;
}

int MMMSSpeed::filter_audio_frames_fast()
{
    int num_frames = avFramesBufferAudio.size();
    for (int i = 0; i < num_frames - 1; i += this->speed_value) {
        AVFrame* frame1 = avFramesBufferAudio[i];
        AVFrame* frame2 = avFramesBufferAudio[i + this->speed_value - 1];

        int num_channels = frame1->channels;
        int num_samples = frame1->nb_samples;

        // Create a new frame with half the samples from frame1 and frame2
        AVFrame* new_frame = create_new_frame(num_samples, num_channels, (AVSampleFormat)frame1->format, frame1->sample_rate);

        // Pointer to new frame data
        float** new_frame_data = (float**)new_frame->data;

        // Process frame1 and frame2
        float** frame1_data = (float**)frame1->data;
        float** frame2_data = (float**)frame2->data;

        for (int ch = 0; ch < num_channels; ++ch) {
            int k = 0;
            for (int j = 0; j < num_samples; j += this->speed_value) {
                new_frame_data[ch][k] = frame1_data[ch][j];
                k++;
            }
            for (int j = 0; j < num_samples; j += this->speed_value) {
                new_frame_data[ch][k] = frame2_data[ch][j];
                k++;
            }
        }

        new_frame->format = frame1->format;
        new_frame->channel_layout = frame1->channel_layout;
        new_frame->channels = frame1->channels;
        new_frame->sample_rate = frame1->sample_rate;
        new_frame->nb_samples = num_samples;

        new_frame->pts = frame1->pts / this->speed_value;
        new_frame->pkt_dts = frame1->pkt_dts / this->speed_value;

        avFramesBufferFilteredAudio.push_back(new_frame);
    }

    // If the number of frames is odd, handle the last frame separately
    if (num_frames % 2 != 0) {
        AVFrame* frame = avFramesBufferAudio[num_frames - 1];
        int num_channels = frame->channels;
        int num_samples = frame->nb_samples;

        // Create a new frame with half the samples from the last frame
        AVFrame* new_frame = create_new_frame(num_samples / this->speed_value, num_channels, (AVSampleFormat)frame->format, frame->sample_rate);

        // Pointer to new frame data
        float** new_frame_data = (float**)new_frame->data;

        // Process the last frame
        float** frame_data = (float**)frame->data;

        for (int ch = 0; ch < num_channels; ++ch) {
            int k = 0;
            for (int j = 0; j < num_samples; j += this->speed_value) {
                new_frame_data[ch][k] = frame_data[ch][j];
                k++;
            }
        }

        new_frame->format = frame->format;
        new_frame->channel_layout = frame->channel_layout;
        new_frame->channels = frame->channels;
        new_frame->sample_rate = frame->sample_rate;
        new_frame->nb_samples = num_samples / this->speed_value;

        new_frame->pts = frame->pts / this->speed_value;
        new_frame->pkt_dts = frame->pkt_dts / this->speed_value;

        avFramesBufferFilteredAudio.push_back(new_frame);
    }
    return 0;
}

void MMMSSpeed::storeAVFrameAudio(AVFrame* input_frame)
{
    AVFrame* temp_frame = av_frame_alloc();
    if (!temp_frame) {
        Logger::Error("Failed to allocate memory for AVFrame", __FILE__, __LINE__);
        return;
    }

    temp_frame->format = input_frame->format;
    temp_frame->channel_layout = input_frame->channel_layout;
    temp_frame->channels = input_frame->channels;
    temp_frame->sample_rate = input_frame->sample_rate;
    temp_frame->nb_samples = input_frame->nb_samples;

    if (av_frame_get_buffer(temp_frame, 0) < 0) {
        Logger::Info("Could not allocate frame data", __FILE__, __LINE__);
        av_frame_free(&temp_frame);
        return;
    }

    if (av_frame_copy_props(temp_frame, input_frame) < 0) {
        Logger::Error("Error Copying Frame Properties.", __FILE__, __LINE__);
        av_frame_free(&temp_frame);
        return;
    }

    if (av_frame_copy(temp_frame, input_frame) < 0) {
        Logger::Error("Error copying frame data: Invalid argument", __FILE__, __LINE__);
        av_frame_free(&temp_frame);
        return;
    }

    temp_frame->pts = input_frame->pts;
    temp_frame->pkt_dts = input_frame->pkt_dts;

    avFramesBufferAudio.push_back(temp_frame);
}


int MMMSSpeed::process_frames(int is_raw_audio_needed, int is_raw_video_needed)
{
    if (adjustVideoPlayback) {
        if (ffmpeg_filter_graph_intialization(this->speed_value) < 0)
        {
            return -1;
        }
    }

    int return_status = 0;
    int64_t buffer_size = 0;
    AVFrame* iframe = nullptr;
    init_input_frame(&iframe);
    AVPacket* input_packet = av_packet_alloc();

    if (is_raw_audio_needed == 1)
    {
        raw_audio_file = fopen(raw_audio_fname.c_str(), "wb");
    }

    if (is_raw_video_needed) {
        if (std::filesystem::create_directory(raw_video_dir)) {
            Logger::Info("Directory created successfully.", __FILE__, __LINE__);
        }
        else {
            Logger::Error("Failed to create directory.", __FILE__, __LINE__);
        }
    }

    /* Write the header of the output file container. */
    if (write_output_file_header()) {
        return_status = -1;
        goto cleanup;
    }

    Logger::Info(fmt::format("Total Frames:{}", total_frames), __FILE__, __LINE__);
    frame_counter = 0;
    mmStats->setProgress(0);

    while (av_read_frame(input_format_context, input_packet) >= 0)
    {
        if (input_format_context->streams[input_packet->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            //Logger::Info(fmt::format("{} TransCode Video Frame.", ++counter), __FILE__, __LINE__);
            ++frame_counter;
            if (transcode_video(iframe, input_packet, is_raw_video_needed) < 0) {
                return_status = -1;
                goto cleanup;
            }
            mmStats->setProgress(static_cast<double>(frame_counter) / total_frames * 100);
            av_packet_unref(input_packet);
        }
        else if (input_format_context->streams[input_packet->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            //Logger::Info(fmt::format("{} TransCode Audio Frame.",++counter2), __FILE__, __LINE__);
            if (transcode_audio(iframe, input_packet, is_raw_audio_needed) < 0) {
                return_status = -1;
                goto cleanup;
            }
            av_packet_unref(input_packet);
        }
    }
    
    counter2 = 0;
    if (adjustAudioPlayback) {
        if (this->playback_type == "fast") {
            filter_audio_frames_fast();
        }

        buffer_size = avFramesBufferFilteredAudio.size();

        for (int64_t i = 0; i < buffer_size; ++i) {
            AVFrame* filt_frame = av_frame_alloc();
            if (!filt_frame) {
                Logger::Error("Failed to allocate memory for AVFrame",__FILE__,__LINE__);
                return -1;
            }

            AVFrame* original_frame = avFramesBufferFilteredAudio.at(i);
            filt_frame->format = original_frame->format;
            filt_frame->channel_layout = original_frame->channel_layout;
            filt_frame->channels = original_frame->channels;
            filt_frame->sample_rate = original_frame->sample_rate;
            filt_frame->nb_samples = original_frame->nb_samples;

            if (av_frame_get_buffer(filt_frame, 0) < 0) {
                Logger::Error("Could not allocate frame data.", __FILE__, __LINE__);
                av_frame_free(&filt_frame);
                return -1;
            }

            if (av_frame_copy_props(filt_frame, original_frame) < 0) {
                Logger::Error("Error Copying Frame properties.", __FILE__, __LINE__);
                av_frame_free(&filt_frame);
                return -1;
            }

            if (av_frame_copy(filt_frame, original_frame) < 0) {
                Logger::Error("Error copying frame data: Invalid argument", __FILE__, __LINE__);
                av_frame_free(&filt_frame);
                return -1;
            }

            //Logger::Info(fmt::format("Encoding Audio Frame {}", ++counter2), __FILE__, __LINE__);
            if (is_raw_audio_needed)
            {
                write_raw_data_audio(filt_frame);
            }
            if (encode_audio(filt_frame) < 0) {
                return_status = -1;
                goto cleanup;
            }
            av_frame_unref(filt_frame);
            av_frame_free(&filt_frame);
        }
    }

    if (encode_video(NULL)) {
        return_status = -1;
        goto cleanup;
    }

    /* Write the trailer of the output file container. */
    if (write_output_file_trailer()) {
        return_status = -1;
        goto cleanup;
    }

cleanup:
    if (iframe)
        av_frame_free(&iframe);
    if (input_packet)
        av_packet_free(&input_packet);
    if (output_codec_context_audio)
        avcodec_free_context(&output_codec_context_audio);
    if (output_format_context) {
        avio_closep(&output_format_context->pb);
        avformat_free_context(output_format_context);
    }
    if (input_format_context) {
        avio_closep(&input_format_context->pb);
        avformat_free_context(input_format_context);
    }
    if (input_codec_context_audio)
        avcodec_free_context(&input_codec_context_audio);
    if (output_codec_context_video)
        avcodec_free_context(&output_codec_context_video);
    if (input_codec_context_video)
        avcodec_free_context(&input_codec_context_video);
    //fclose(outputFile);
    if (raw_audio_file)
        fclose(raw_audio_file);
    return return_status;
}

int MMMSSpeed::transcode_video(AVFrame* iframe, AVPacket* input_packet, int is_raw_needed)
{
    if (!adjustVideoPlayback) {
        return MMMMSConversion::transcode_video(iframe, input_packet, is_raw_needed);
    }
    char fileNameBuffer[1024];
    AVFrame* filt_frame = av_frame_alloc();
    if (!filt_frame) {
        Logger::Error("Could not allocate video frame", __FILE__, __LINE__);
        return -1;
    }
    int response = avcodec_send_packet(input_codec_context_video, input_packet);
    if (response < 0) {
        Logger::Error("Error while sending packet to decoder:", __FILE__, __LINE__);//std::string(av_err2str(response)) << endl;
      return response;
    }
  
    while (response >= 0) {
      response = avcodec_receive_frame(input_codec_context_video, iframe);
      if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
        break;
      } else if (response < 0) {
        Logger::Error("Error while receiving frame from decoder:",__FILE__,__LINE__); //std::string(av_err2str(response)) << endl;
        return response;
      }

      if(is_raw_needed == 1)
      {
        ++rawCounter;
        std::string file_path_pgm = raw_video_dir + "/" + "pgm-" + std::to_string(rawCounter) + ".pgm";
        snprintf(fileNameBuffer, sizeof(fileNameBuffer), file_path_pgm.c_str());
        pgm_save(iframe->data[0], iframe->linesize[0],iframe->width, iframe->height, fileNameBuffer);
      }
      iframe->pts = iframe->best_effort_timestamp;
      if (av_buffersrc_add_frame_flags(buffersrc_ctx, iframe, AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
          Logger::Error("Error while feeding the filter graph", __FILE__, __LINE__);
          break;
      }              

      while (1) {
          response = av_buffersink_get_frame(buffersink_ctx, filt_frame);
          if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
            break;
          if (response < 0)
            return response;
          filt_frame->pts = av_rescale_q(filt_frame->pts, av_buffersink_get_time_base(buffersink_ctx), output_format_context->streams[videoStreamIndex]->time_base);
          if (encode_video(filt_frame)) return -1;
      }
      av_frame_unref(iframe);
      av_frame_unref(filt_frame);
    }

    if(filt_frame)
        av_frame_free(&filt_frame);
    return 0;
}
