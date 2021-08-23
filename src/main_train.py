import tensorflow as tf
from data.generator import DataGenerator
import os
import datetime
import string
from network.model import HTRModel
from data import preproc as pp
from data import evaluation
import cv2

device_name = tf.test.gpu_device_name()

if device_name != "/device:GPU:0":
    raise SystemError("GPU device not found")

print("Found GPU at: {}".format(device_name))


possible_sources=["bentham", "iam", "rimes", "saintgall", "washington"]
possible_archs=[  "florAT"]#  "flor" ,"bluche", "puigcerver","puigcerver_octconv","florPP","florBN"

# define parameters
source = possible_sources[2]
epochs = 1000
batch_size = 16
for arch in possible_archs:


    # define paths
    source_path = os.path.join("..", "data", f"{source}.hdf5")
    output_path = os.path.join("..", "output", source, arch)
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
    os.makedirs(output_path, exist_ok=True)

    # define input size, number max of chars per line and list of valid chars
    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]

    print("source:", source_path)
    print("output", output_path)
    print("target", target_path)
    print("charset:", charset_base)

    dtgen = DataGenerator(source=source_path,
                          batch_size=batch_size,
                          charset=charset_base,
                          max_text_length=max_text_length)

    print(f"Train images: {dtgen.size['train']}")
    print(f"Validation images: {dtgen.size['valid']}")
    print(f"Test images: {dtgen.size['test']}")

    # create and compile HTRModel
    model = HTRModel(architecture=arch,
                     input_size=input_size,
                     vocab_size=dtgen.tokenizer.vocab_size,
                     beam_width=10,
                     stop_tolerance=20,
                     reduce_tolerance=15)

    model.compile(learning_rate=0.001)
    model.summary(output_path, "summary.txt")

    # get default callbacks and load checkpoint weights file (HDF5) if exists
    model.load_checkpoint(target=target_path)

    callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

    # to calculate total and average time per epoch
    start_time = datetime.datetime.now()

    h = model.fit(x=dtgen.next_train_batch(),
                  epochs=epochs,
                  steps_per_epoch=dtgen.steps['train'],
                  validation_data=dtgen.next_valid_batch(),
                  validation_steps=dtgen.steps['valid'],
                  callbacks=callbacks,
                  shuffle=True,
                  verbose=1)

    total_time = datetime.datetime.now() - start_time

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    min_val_loss = min(val_loss)
    min_val_loss_i = val_loss.index(min_val_loss)

    time_epoch = (total_time / len(loss))
    total_item = (dtgen.size['train'] + dtgen.size['valid'])

    t_corpus = "\n".join([
        f"Total train images:      {dtgen.size['train']}",
        f"Total validation images: {dtgen.size['valid']}",
        f"Batch:                   {dtgen.batch_size}\n",
        f"Total time:              {total_time}",
        f"Time per epoch:          {time_epoch}",
        f"Time per item:           {time_epoch / total_item}\n",
        f"Total epochs:            {len(loss)}",
        f"Best epoch               {min_val_loss_i + 1}\n",
        f"Training loss:           {loss[min_val_loss_i]:.8f}",
        f"Validation loss:         {min_val_loss:.8f}"
    ])

    with open(os.path.join(output_path, "train.txt"), "w") as lg:
        lg.write(t_corpus)
        print(t_corpus)

    start_time = datetime.datetime.now()

    # predict() function will return the predicts with the probabilities
    predicts, _ = model.predict(x=dtgen.next_test_batch(),
                                steps=dtgen.steps['test'],
                                ctc_decode=True,
                                verbose=1)

    # decode to string
    predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
    ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]

    total_time = datetime.datetime.now() - start_time

    # mount predict corpus file
    with open(os.path.join(output_path, "predict.txt"), "w") as lg:
        for pd, gt in zip(predicts, ground_truth):
            lg.write(f"TE_L {gt}\nTE_P {pd}\n")
       
    for i, item in enumerate(dtgen.dataset['test']['dt'][:10]):
        print("=" * 1024, "\n")
        print(ground_truth[i])
        print(predicts[i], "\n")


    evaluate = evaluation.ocr_metrics(predicts, ground_truth)

    e_corpus = "\n".join([
        f"Total test images:    {dtgen.size['test']}",
        f"Total time:           {total_time}",
        f"Time per item:        {total_time / dtgen.size['test']}\n",
        f"Metrics:",
        f"Character Error Rate: {evaluate[0]:.8f}",
        f"Word Error Rate:      {evaluate[1]:.8f}",
        f"Sequence Error Rate:  {evaluate[2]:.8f}"
    ])

    with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
        lg.write(e_corpus)
        print(e_corpus)