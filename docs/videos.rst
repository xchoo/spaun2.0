******
Videos
******

The following videos show Spaun performing cognitive tasks.
Some of these videos are referenced in the original Science paper
by their identifiers (e.g., movie S1).

Introduction
============

.. topic:: Introduction to Spaun (1)

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/dKaqFz_WoIw" frameborder="0" allowfullscreen></iframe>

.. topic:: Introductory to Spaun (2)

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/P_WRCyNQ9KY" frameborder="0" allowfullscreen></iframe>

.. topic:: Try the Spaun tasks

   Examples of the tasks that Spaun performs.
   Here they are show with the same timing
   and in the same format (images on a background) as they are shown to Spaun.
   You can perform the tasks like Spaun
   by writing your answers on a piece of paper.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/IrqK_fA8ozE" frameborder="0" allowfullscreen></iframe>

.. topic:: Spaun performing several tasks

   Only neural activity is shown (no decoding of the activity).
   The model is running at about 1/2 real time.
   The tasks it performs are:
   A1 (recognition), A3 (serial working memory), A7 (syntactic pattern induction).
   The tasks are described in the videos of each separate task
   (see the next section).

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/0W9GP8QJHfA" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      The purpose of this video is to allow you to watch
      just the input, output, and neural activity of Spaun
      while it performs several tasks.
      These tasks are demonstrated and described in more detail in a previous video.

      Here we can see that the input is a 28x28 pixel image on the right,
      which is processed by the spiking neural networks of the model.
      Activity of some of these networks is shown
      by the coloured plots on the brain.
      Red indicates high activity, and blue indicates low activity.
      As the model has 2.5 million neurons, only a small proportion are shown.

      However, these are mapped the their corresponding anatomical areas.
      For example, inferotemporal cortex (IT),
      the highest level of the visual hierarchy is a the back of the brain,
      while motor areas are in the vertical stripe in the middle.
      Executive control areas are at the front,
      with working memory areas just behind them in prefrontal cortex (PFC).

      In addition, two parts of the basal ganglia (BG),
      which in fact lies underneath cortex,
      are shown in the horizontal stripes in the middle.
      The top stripe is the striatum (Str), the input to BG,
      the bottom stripe is globus pallidus internus (GPi), output from BG.
      The BG as a whole is monitoring cortical states
      to determine the next appropriate cognitive or physical action.

      Physical actions themselves are evident in the movement of the arm.
      Though simplified, the arm is a dynamic, physical model of a limb,
      having mass, length, inertia, and so on.

      The simulation is likely too fast and unfamiliar
      to demonstrate the subtleties of the model.
      In the next video, I show similar tasks,
      but include graphs that allow us to see
      dynamic interpretations of this neural activity
      at a slower speed -- essentially,
      we can read Spaun's mind to get a sense of how it thinks.

.. topic:: How Spaun performs several tasks

   Spaun performing several tasks.
   Neural activity and decoding of that activity (thought bubbles) are shown.
   The model is running at about 1/6 real time.
   The tasks it performs are:
   A1 (recognition) three times, A0 (copy drawing) twice,
   A4 (counting), A5 (question answering), A3 (serial working memory).
   The tasks are described in the videos of each separate task
   (see the next section).

   * Input: 28x28 images (on the right)

   * Output: movements of a physically modeled arm
     (top down view in the top right corner)

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/RrxmlbZa7C4" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      In this video, we see not only
      the activity of the various brain areas as in the previous video,
      but also decodings of the neural representations in those areas.
      The first task we are watching is a recognition task.
      Spaun must indicate with its arm what input image it saw.
      The images are randomly picked from a publicly available database
      of ~60,000 examples of human hand written digits.
      The classification it thinks it saw is indicated
      by the decoding of the neural activity in the thought bubble
      at the back of the brain.
      A subset of the voltage spikes generated by individual neurons
      is shown scrolling through the bubble.
      The decoded value itself is displayed on top of that neural activity.

      The thought bubble at the top of the brain
      is a decoding of the representation in motor cortex.
      When this activity is decoded, a series of target points
      to move the arm to is generated.
      Movement of the arm through these points is controlled
      by the rest of the motor hierarchy below this level of representation.
      The resulting motion is evident in the movement of the arm itself.

      Here the model is viewing a 2,
      which we can see is represented in the activity in visual areas,
      and which drives motor areas to generate
      a series of targets that move the arm
      in Spaun's default 'handwriting'.

      The next task is slightly different.
      It is called Copy Drawing.
      Instead of just recognizing the digit,
      the model must attempt to reproduce
      the visual properties of the image it saw.
      As you can see, the first example it is shown
      is a two with a looped bottom.
      The model then produces a two with a similar characteristic loop.
      In the next example it is shown,
      it must perform the same task,
      but the two that is displayed has a straight bottom.
      As we can see from the motor activity and ultimately the arm motion,
      the representations in Spaun are able
      to capture this subtle visual difference.
      These representations are thus useful not only for categorization,
      but also for encoding fine perceptual properties.

      The next task is a counting task.
      This task demonstrates that Spaun
      can not only represent and categorize numbers,
      it also has an understanding of numerical concepts,
      and particularly their progression relations.
      Spaun is first shown a number to start counting from (4),
      and then it is shown how much to count by (3).

      Several new areas of the brain are shown in this example.
      In particular, three working memory areas
      are displayed right behind the front most area.
      If you pay attention to the thought bubbles above the working memory areas,
      you will see how Spaun performs this task.
      The topmost working memory stores the starting number.
      The second stores the number of counts,
      and the third stores how many counts have been performed.
      You can watch as Spaun increases the starting number
      until it has counted the correct number of times.
      It then generates the appropriate motor response.

      Next we have a question answering task.
      Here, Spaun memorizes a list of numbers
      and then answers a query about the list.
      You can watch the topmost working memory
      as it stores the items in the order they are shown.
      The frontmost thought bubble shows what task
      Spaun thinks it is performing (A means answering a question),
      and what its current goals are.
      Essentially, this is a memory of the current task context.
      Once the list of numbers is complete, Spaun is asked a question.
      The kind of question is indicated by a letter
      (here a P, which indicates a position question).
      The next digit indicates which position is being queried.
      So here Spaun is asked "what is in position 5".
      Spaun answers the question by decoding its own working memory,
      and using the result to drive the arm.
      You will notice that the front bubble indicates 'dec'
      because Spaun is decoding memories to generate arm movements.

      This final task is a simple serial working memory task
      in which Spaun must memorize a list of items in order,
      and then repeat it back.
      We have seen several tasks that Spaun performs correctly,
      but the errors it makes are equally important to determining
      if it is a good model of human cognition.
      As you may know, people make more errors while recalling longer lists.
      In fact, the pattern of errors in human memory
      is quite stereotypical -- we tend to remember
      the beginnings and ends of lists better than the middle.
      As we watch Spaun encode this list,
      I should note that the darkness of the letter
      in the thought bubble indicates
      how well that item can be decoded from the current neural representation.
      As you can see, the 8 in the middle of this list is beginning to fade,
      indicating that it is being forgotten by the model.
      Spaun is now drawing its memory of the list,
      but when it gets to the 8,
      it is no longer confident in its own decoding of that memory,
      so it draws a horizontal line, indicating it has no answer.
      It then proceeds to complete the list,
      as it does remember subsequent items.
      If we look at the pattern of errors over many such runs of Spaun,
      we will notice that it matches the details of human error patterns.

      I should note that, like humans,
      Spaun does not change its brain in between performing tasks.
      Perhaps surprisingly,
      this is not true of many contemporary cognitive models.
      We have now seen about half of the tasks Spaun can perform.
      The rest are in other videos on the Nengo.ca website.

Tasks
=====

.. topic:: Copy drawing (S1)

   Spaun performing copy drawing.
   The model is presented the image,
   and then must try to reproduce the visual features
   of the presented image from memory.
   The top video shows Spaun reproducing a '2' with a looped bottom.
   The bottom video shows Spaun reproducing a '2' with a straight bottom.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/WNnMhF7rnYo" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      For this copy drawing task,
      the Spaun model must reproduce the perceptual features
      of the input using its motor system.
      Here, both of the inputs are the number 'two',
      but they are drawn in two different styles.
      The first has a looped bottom, and the second a straight bottom.
      Spaun's ability to capture these differences
      with its motor responses shows that its neural representations
      carry deep semantic features of its input.

.. topic:: Recognition (S2)

   Spaun performing recognition.
   Spaun must categorize the presented visual input.
   The images are taken from the publicly available MNIST database.
   Overall the model has 94% accuracy
   (people have about 98% accuracy on this data set).

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/f6Ul5TYK5-o" frameborder="0" allowfullscreen</iframe><

   .. container:: toggle

      .. container:: header

         **Transcript**

      Spaun performs this simple recognition task three times in a row.
      The thought bubble at the back of the brain
      shows the decoding of the neural activity
      at the highest level of the visual hierarchy.
      As you can see, this is quite accurate,
      and only very briefly delayed from the input stimulus.

      The activity of this top-level,
      which we can think of as inferotemporal cortex,
      is at the end of a four layer hierarchy
      that includes visual areas V1, V2, and V4.
      The neurons in the earliest visual area
      (V1, also called primary visual cortex)
      have receptive fields and neural responses like those of primates.
      However, these areas are not shown in the video.

      Although the images Spaun is shown are quite variable,
      being examples of human hand-writing,
      the model is about 94% accurate in recognizing digits.
      This is only slightly below
      the 98% accuracy of humans on the same data set.
      It is almost 100% accurate on the images
      of type-written digits used to specify tasks.

      The neural activity in the motor area
      is at the top of the motor hierarchy.
      This can be thought of as
      a low dimensional representation of a motor plan,
      that is made progressively higher-dimensional
      as it proceeds down the hierarchy.
      The plan needs to become higher dimensional
      in order to control the many muscles
      that ultimately drive the arm.

      This task demonstrates that Spaun's neural representations
      not only allow successful categorization
      of naturally varying stimuli,
      but also allow that categorization
      to drive appropriate behaviour.

.. topic:: Reinforcement Learning (S3)

   Spaun performing reinforcement learning.
   This is a 3-armed bandit task.
   Spaun generates a number between 0-2,
   then is provided a reward or not,
   indicated by a 1 or 0 respectively.
   The reward is given with
   a probability of .12 for 'bad' actions
   and .72 for 'good' actions.
   In the video shown here, the 'good' action is choosing a 2.
   In longer runs, the 'good' action switches every once in a while.
   This task demonstrates that Spaun
   can change its behavior based on
   probabilistic rewards from the environment.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/vuGDYajWyhU" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      Task number two is a reinforcement learning task.
      After each question mark,
      Spaun must guess the 'best' number between zero and three.
      The best number is the number that generates the most reward.
      In the simulation, a positive reward is indicated by a 1,
      and a lack of reward is indicated by a 0.
      However, even the best number is only probabilistically rewarded.
      Such tasks are called 'bandit tasks'
      because they are reminiscent of the chance rewards
      received from one-armed bandits at casinos.

      As you can seen in this simulation,
      Spaun begins with several guesses
      that do not generate much reward,
      until it has determined that 'two' is the best value.
      Spaun guesses two several times in a row.
      Eventually the two is not rewarded,
      but Spaun continues to guess that value.
      Soon after, the two is not rewarded a second time.
      So it changes its guess.

      The detailed spiking patterns of
      the ventral striatum in Spaun are strikingly similar
      to those of rats performing the same kind of bandit task.

.. topic:: Serial Working Memory (S4)

   Spaun performing a serial working memory task.
   The model has to repeat back the list of digits in order.
   The top video shows a correct performance.
   The bottom video shows an error halfway through the list.
   Spaun displays primacy and recency effects in serial working memory,
   similarly to people.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/XxIzmkWygjY" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      A3 indicates that Spaun must perform a serial working memory task.
      This task consists of memorizing a list of numbers,
      and then repeating the list back.
      The working memory encoding can be seen
      in the second thought bubble from the front.
      Spaun has no trouble with short lists, just like people.

      However, the second time Spaun performs this task,
      it is confronted with a much longer list.
      Watch the working memory thought bubble
      to see that items in the middle of the list
      begin to fade as they become more difficult
      to decode from the neural spikes.
      As you watch Spaun write out its answer,
      keep in mind that it draws a horizontal line
      to indicate when it doesn't know the item.
      Notice that even though it forgot an item
      in the middle of the list,
      it can complete the list, again just like people.
      Running many instances of Spaun
      shows that it generates errors when recalling lists
      that match well to human subject data.

.. topic:: Counting (S5)

   Spaun performing silent counting.
   The model is shown a starting number,
   and then a number to count to.
   It internally counts from the starting number
   the designated number of counts, and writes the response.
   This is much like an adding task.
   The model shows Weber's law effects on this task
   (scaling of response time variance with mean),
   and has reaction times consistent with human data.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/mP7DX6x9PX8" frameborder="0" allowfullscreen</iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      In the fourth task, Spaun counts silently.
      Here it starts at 4, and counts 3 positions.
      You can watch the silent counting
      by noticing that the working memory starting at four
      increments by one three times.
      When humans perform this task,
      the variability of their reaction time
      increases with the number of counts.
      This kind of relationship is called Weber's law in psychology,
      and is evident in Spaun's behaviour as well.

.. topic:: Question Answering (S6)

   Spaun performing question answering.
   The model is shown a string of digits.
   It is then shown either a 'P' or a 'K', followed by a digit.
   P indicates that it must report the item
   in the position indicated by the digit.
   K indicates it must report which position the given digit is in.
   The model shows primacy and recency effects on this task
   (regardless of question type),
   predicting that people would do the same
   (though this is as yet untested).

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/pPPXncTBv4o" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      The fifth task is a question answering task.
      Spaun is shown a list of numbers,
      and is then asked a question about the list.
      There are K questions and P questions.
      Here, it is asked a P question,
      which means it must indicate
      what is in the position number provided
      (here it's asked what is in position 5).
      This task demonstrates that Spaun has flexible, rapid access
      to information that it encodes.

.. topic:: Rapid Variable Creation (S7)

   Spaun performing rapid variable creation.
   The model is shown several input/output examples.
   It is then given an input and must produce the output.
   The model has to figure out the pattern
   in the presented input/output examples
   in order to solve the task.
   This requires induction,
   and more specifically,
   syntactic generalization.
   Both of these features have been argued
   to be central to human cognition.
   Several authors have argued
   that no neural models can do this task
   without implementing a classical architecture
   (Hadley, 2009; Fodor & Pylyshyn, 1988; Marcus, 2001; Jackendoff, 2002).
   Spaun does not implement a classical architecture,
   but can perform the task.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/tPRbphzQ-T8" frameborder="0" allowfullscreen></iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      It has been argued by several researchers
      that neural models of cognition
      cannot explain a basic feature of human behaviour:
      the ability to rapidly generalize
      over syntactically structured input.
      Task 6, shown here, demonstrates that Spaun
      is able to perform such generalization.
      It is shown a series of input/output pairs
      that bear some relation to one another.
      Input is stored in one memory (here, 0014),
      and output is stored in another (here 14).
      After having seen three examples,
      Spaun must inductively determine
      what the relationship is -- and it must do this
      as quickly as people do.
      To determine if it is successful,
      Spaun is provided an input
      it hasn't seen before (here 0074).
      As shown, it responds correctly,
      suggesting that it has figured out
      the underlying structure of the examples.

.. topic:: Fluid Reasoning (S8)

   Spaun performing a task based on the Raven's Progressive Matrices.
   This is a task of fluid intelligence.
   Spaun must watch input that provides
   two examples of a pattern over numbers.
   It must then complete the third pattern over numbers.
   The patterns come in sets
   of three [ ] brackets,
   e.g. [A][B][C] is the first pattern.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/qcZe-2eWaeM" frameborder="0" allowfullscreen</iframe>

   .. container:: toggle

      .. container:: header

         **Transcript**

      Task 7 is analogous to a reasoning task
      found on the Raven's Progressive Matrix
      test of fluid intelligence.
      This test is one of the most common tests of IQ.
      As in the previous rapid variable creation task (task 6),
      Spaun must figure out the pattern
      over structured input given some examples.
      In this case, the examples are
      two sets of three number patterns
      (for instance, at the moment it is being shown 4, 44, 444).
      Spaun has to figure out the underlying pattern,
      and apply it to complete a new set
      given only the first two parts of the pattern.
      One important feature of this task
      is that no new components are introduced
      compared to other tasks it can do.
      This shows that Spaun can flexibly
      redeploy the cognitive resources
      it has to solve a wide variety of challenges.

.. topic:: WM 3-link Arm (bonus)

   Spaun performing serial working memory,
   but with a more sophisticated arm.
   This arm has three joints (instead of two),
   and movement is controlled by muscle contractions only (6 muscles).
   The amount of tension in a muscle is shown by its colour.

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/FEEEoodC6Xc" frameborder="0" allowfullscreen></iframe>
