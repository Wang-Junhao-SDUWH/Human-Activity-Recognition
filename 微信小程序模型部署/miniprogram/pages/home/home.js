// pages/home/home.js
const db = wx.cloud.database();
const myaudio = wx.createInnerAudioContext();
myaudio.autoplay = true;

/*
  踩过的坑
  · 留意JavaScript为“整数”赋浮点数不会自动截断小数，须手动parseInt
  · 通过修改JS的data实现对DOM的操作，须使用setData()才能使修改生效
*/
/*
  整体思路
    实时播报
      点击按钮后开始30s持续的数据采集，采集的数据实时存放到this.data中，实现共用
      每隔x秒进行一次实时播报，从this.data取出最新的一段数据进行识别
    计数
      30s数据采集结束后，取出全部数据，按照动作类别进行分段，每段分别计数
      计数方法为先对序列施加三次中值滤波，然后根据极大值、极小值的个数算出运动的次数
*/
Page({

  /**
   * 页面的初始数据
   */
  data: {
      motion:["大腿内侧动态拉伸","缓冲深蹲跳","臀部动态拉伸","早安式体前屈"],
      audiosrc:["cloud://rustic-26.7275-rustic-26-1302385966/audio/0.mp3","cloud://rustic-26.7275-rustic-26-1302385966/audio/1.mp3","cloud://rustic-26.7275-rustic-26-1302385966/audio/2.mp3","cloud://rustic-26.7275-rustic-26-1302385966/audio/3.mp3"],
      imgsrc:["cloud://rustic-26.7275-rustic-26-1302385966/img/大腿内侧动态拉伸.gif",
              "cloud://rustic-26.7275-rustic-26-1302385966/img/缓冲深蹲跳.gif",
              "cloud://rustic-26.7275-rustic-26-1302385966/img/臀部动态拉伸.gif",
              "cloud://rustic-26.7275-rustic-26-1302385966/img/早安式体前屈.gif"],
      isDisabled:false,
      rows:[279,251,396,374],
      mean: [
        [0.03002617851003559,0.1278009750516398,0.3217516880364198,0.808791933518853,-0.583730977378815,16.084380610412925,1.3925229108976658,-0.051829359515327376,0.1953539638464306,0.2958563399587733,0.46427772646319615,-0.8499019861400365,15.572710951526032,1.314179712603232,-0.009483878292347798,0.043752905698930235,0.2006418441524752,0.38488276156193907,-0.45663259355475794,16.236983842010773,0.8415153551166973,-0.9081630021016615,0.025716627449749795,0.15348386753565874,-0.6092380526211844,-1.2150308543985662,0.0,0.6057928017773793,0.34762740892975913,0.016779600211568275,0.12116401770062585,0.5782984592639131,0.11950854977324962,31.590664272890486,0.45878990949066384,-0.023284957439614225,0.012385260384114574,0.10148935438401423,0.17979795783572705,-0.24678154374865327,14.217235188509875,0.42657950158438024],[0.003019158294065525,0.2187526048041284,0.41743560297933024,0.8658580120159683,-1.0694660340718558,16.98802395209581,1.9353240460878252,-0.001122952410298474,0.09863200685384271,0.28393336230441807,0.6398028996806393,-0.6508022665548907,15.942115768463074,1.2906051662355285,-0.0023839369718831683,0.24400250200924153,0.43373921540253413,1.0561752959680644,-0.9610312381636728,15.46506986027944,2.017206534131737,0.02672192513107898,0.028340431725953814,0.15318206770374782,0.3932117487824351,-0.33165647867864234,18.127744510978044,0.724868227461077,-0.9753022464128817,0.46828268058801337,0.6529632139492071,-0.07770945082315382,-2.48913341676647,1.1976047904191616,2.4114239659433148,-0.035295215121482135,0.028241058903107352,0.14672790763267293,0.29962412404111843,-0.40370228048503,13.347305389221557,0.7033264045261476],[0.03229432630936823,0.45181223036165064,0.6485541276137303,1.4953982454355852,-1.5813878747607364,16.998773006134968,3.076786120196316,0.03222153838465429,0.3845136311945791,0.5946318400946317,1.5519769006134985,-1.3179141943558286,16.536196319018405,2.869891094969322,-0.07220947634583091,1.8435228744651666,1.3197140354009478,2.2229979373999953,-2.5582583163337467,15.25398773006135,4.78125625373374,-0.9289687457851855,0.13711191592466168,0.3614985127933906,-0.22340546080453974,-1.746387924417179,0.16441717791411042,1.5229824636126348,-0.041463265034341334,0.13010736649156476,0.3500227657330913,0.5458488726512881,-0.7656273799888346,16.121472392638037,1.3114762526401218,0.1622319678450004,0.05273832941219914,0.21392194955420202,0.8259755473668708,-0.29492314581190165,26.12883435582822,1.1208986931787734],[-0.004964549894444884,0.11330214337130351,0.31506750445344023,0.5829095257727644,-0.5803245910577716,13.506189821182943,1.163234116830535,0.032977835740997455,0.5714647086316469,0.7097119110600872,1.1735607762382387,-1.1736998518250348,17.91884456671252,2.3472606280632724,0.013359176790655809,0.2936014170578623,0.5101568049015229,0.7940311539093536,-0.8623247579299591,18.174690508940852,1.6563559118393114,0.47787081841494133,0.08241080831074904,0.26180679437678606,0.8541893446079772,0.05174140878583219,26.4525447042641,0.8024479358221462,-0.19177032189955276,0.08520170077248508,0.265952597864117,0.17886675290013734,-0.6609190117145797,15.042640990371389,0.8397857646147178,0.5401580364219407,0.0714888072825026,0.2377386027195113,0.9778821241540581,0.21103539283301231,31.902338376891333,0.7668467313210453]
      ],
      stdev: [
        [0.09282049202805794,0.17922557774365866,0.15595027942906756,0.5912641719575077,0.26419080872093753,3.4088949746802597,0.7030523798736482,0.17075511990297457,0.5209828019635859,0.32865927066424216,0.33632561602548194,1.2871288188460797,3.2393447929918575,1.4173969281861454,0.06614922641411768,0.02654919201398659,0.059178065236137,0.14449261046643533,0.19976452575792647,4.46181201150356,0.25617826636815877,0.06340803232081996,0.016193139065941713,0.046510359521698054,0.12401895443851034,0.11128567301317381,0.0,0.17520263987254914,0.12300179774005739,0.013031681667320634,0.045854727041878755,0.19553723507627652,0.11493873135917809,1.2499434909666063,0.16283452088425723,0.07420764209166253,0.014175493807637295,0.045704722283146614,0.08320204031937262,0.1532583561397577,7.167913433458229,0.16218510344258757],[0.0599595502631046,0.21538445198021106,0.2111613658171164,0.4767352279414757,0.6426797018852477,2.5541840746949207,1.0074711752513479,0.10172960132940606,0.13750157294304996,0.1343498430575929,0.34050535865719866,0.4298103653960271,3.3362617874757445,0.6190537051510968,0.05674272553203606,0.26306697825325903,0.23661052517015838,0.702873801510053,0.6211540481117478,2.853993245456228,1.208804588240182,0.07002132654717719,0.025681260027993084,0.06989590282468759,0.24756123328358665,0.19778829338535642,5.267224003457116,0.35957062023881214,0.12281094901945488,0.2429465234280563,0.20495259271540978,0.1829739540693586,0.538359831365315,1.900230052256043,0.6410764175420995,0.05625424337724182,0.03333796472920668,0.08200856043666886,0.19134399545947314,0.3032316193511722,4.84305045653572,0.42108957869268554],
        [0.15222106735636745,0.2474872614701667,0.1767147152684749,0.5973170257882129,0.7407878593890653,4.0641355181434955,1.0898127746936812,0.11291833524972107,0.230150138648286,0.1759676087058111,0.77468140612206,0.5696287336947413,3.611454988762651,1.1716191192724457,0.7536428869197108,0.8798032696751807,0.3193789225335872,1.180884895658492,1.386116934911039,6.587758036372859,1.2765251037562537,0.09778325743772226,0.05538549826582941,0.08024114497101513,0.23622890866466978,0.27011084805658436,0.42635547418550673,0.38420407371323917,0.22412409856618754,0.059883628762985426,0.08718231495638352,0.20291548385145267,0.31524878024765207,7.555118855797023,0.32987687271746097,0.08589853796080901,0.0467381904297207,0.08357211617917605,0.4434934847139987,0.27033971259097495,4.555887245143847,0.6177740977910967],
        [0.253376570471323,0.08568116420608411,0.11854932487748472,0.4305866753553164,0.32280335517642483,8.739867974509215,0.4165260221061458,0.6609854313562766,0.4169131196884567,0.26051307865705303,0.736190210562934,0.9066814265528165,10.123358407168768,0.856380492321561,0.47389342963429243,0.2080976336584102,0.1827221287496823,0.4656194136915886,0.6511695250997415,9.982685720522161,0.5755060349827789,0.29617522281061653,0.06355729044835125,0.11784359410698976,0.17614242422082,0.2863661617509351,7.660326111376754,0.2878948017017142,0.27947619605469876,0.06264722071084539,0.12037794172369416,0.1514925761849343,0.3227978167479814,10.296143741016477,0.3075001934462173,0.237923419359211,0.06158625722864094,0.12243276825570232,0.3200415347429671,0.1540788595099389,0.5116629087351562,0.3377243289293871]
        ],
      axis_map:[3,4,2,2],
      items: [
        {
          "txt":"大腿内侧动态拉伸",
          "bgc":"#FFFFFF",
          "color":"#32B67A",
          "borc":"#32B67A"
        },
        {
          "txt":"缓冲深蹲跳",
          "bgc":"#32B67A",
          "color":"#FFFFFF",
          "borc":"#FFFFFF"
        },
        {
          "txt":"臀部动态拉伸",
          "bgc":"#FFFFFF",
          "color":"#32B67A",
          "borc":"#32B67A"
        },
        {
          "txt":"早安式体前屈",
          "bgc":"#32B67A",
          "color":"#FFFFFF",
          "borc":"#FFFFFF"
        },
      ]
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
      wx.setInnerAudioOption({ // ios在静音状态下能够正常播放音效
        obeyMuteSwitch: false,   // 是否遵循系统静音开关，默认为 true。当此参数为 false 时，即使用户打开了静音开关，也能继续发出声音。
      });
  },
  
  /*
    顶层函数，实现对整体时序的控制
  */
  record:function(e){
    this.data.gyr_x = [];
    this.data.gyr_y = [];
    this.data.gyr_z = [];
    this.data.acc_x = [];
    this.data.acc_y = [];
    this.data.acc_z = [];
    this.data.t = [];
    var begin;

    /*播报采集音频*/
    myaudio.src="cloud://rustic-26.7275-rustic-26-1302385966/audio/开始.mp3";
    /*禁用<开始测试>按钮，防止误触*/
    this.setData({isDisabled : true});

    //点击按钮2s后开始采集数据
    setTimeout(()=>{
      wx.showToast({
        title: '计时开始',
        icon: 'success',
        duration: 2000
      });
      var that = this.data;
      wx.startGyroscope({
        interval:"game"
      });
      wx.startAccelerometer({
        interval:"game",
        success: function(){
          wx.onAccelerometerChange(function(result) {
            that.gyr_x.push(result.x);
            that.gyr_y.push(result.y);
            that.gyr_z.push(result.z);
          });
          wx.onGyroscopeChange(function(result){
            that.acc_x.push(result.x);
            that.acc_y.push(result.y);
            that.acc_z.push(result.z);
            that.t.push((new Date().getTime()) - begin);
          })
          begin = new Date().getTime();
        }
      })
    },2000);

    /*实时播报*/
    //开始采集数据2s后开始以3s为间隔实时播报
    setTimeout(()=>{
      this.data.interval1 = setInterval(()=>{
        //从页面data中取出最新的32个数据点
        var seq = new Array(6);
        seq[0] = this.data.acc_x.slice(-32);
        seq[1] = this.data.acc_y.slice(-32);
        seq[2] = this.data.acc_z.slice(-32);
        seq[3] = this.data.gyr_x.slice(-32);
        seq[4] = this.data.gyr_y.slice(-32);
        seq[5] = this.data.gyr_z.slice(-32);
        var feature = this.extract_features(seq);
        var prob = this.naive_calc_prob(feature);
        //motion将动作序号映射为动作字符串
        var motion = this.data.motion;
        var index = 0;
        var maxi = -1;
        for(let i=0;i<prob.length;i++){
          if(prob[i]>maxi){
            maxi = prob[i];
            index = i;
          }
        }
        //播放相应动作的音频
        myaudio.src=this.data.audiosrc[index];
      },3000);
    },2000);

    /*结束数据采集*/
    //开始采集数据28s后停止实时播报（数据仍在采集）
    setTimeout(()=>{
      clearInterval(this.data.interval1);
    },28000 + 2000)
    //开始采集数据29s后播报结束语音
    setTimeout(()=>{
      myaudio.src="cloud://rustic-26.7275-rustic-26-1302385966/audio/结束.mp3";
    },29000+2000)
    //开始采集数据30s后停止数据采集
    setTimeout(()=>{
      this.data.tower = this.count();
      wx.stopAccelerometer({
        complete: (res) => {},
      });
      wx.stopGyroscope({
        complete: (res) => {},
      });
      this.setData({isDisabled : false});
      db.collection("SF").add({
        data:{
        "acc_z":this.data.acc_z,
        "t":this.data.t
        }
      });
      wx.showToast({
        title: '计时结束',
        icon: 'success',
        duration: 1000
      });
      //停止数据采集2s后显示计数信息
      setTimeout(()=>{
        wx.showModal({
          title:"运动次数统计",
          content:"大腿内侧动态拉伸："+this.data.tower[0]+"\n"+
                  "缓冲深蹲跳："+this.data.tower[1]+"\n"+
                  "臀部动态拉伸："+this.data.tower[2]+"\n"+
                  "早安式体前屈："+this.data.tower[3]+"\n",
        })
      },2000);
    },30000 + 2000);
  },
  /*从目标时间序列提取时域特征*/
  extract_features:function(seq){
    var feature = [];

    /*时域信息提取*/
    /*六轴逐轴提取*/
    for(let i=0;i<6;i++){
      var m=0;
      for(let j=0;j<seq[i].length;j++){
        m+=seq[i][j];
      }
      m /= seq[i].length;
      feature.push(m);

      var variance = 0;
      for(let j=0;j<seq[i].length;j++){
        variance += (seq[i][j] - m)*(seq[i][j] - m);
      }
      variance /= seq[i].length;
      feature.push(variance);
      feature.push(Math.sqrt(variance));

      var max = -1;
      var min = 100000;
      var overzero = 0;
      for(let j=0;j<seq[i].length;j++){
        if(seq[i][j]>max)
          max = seq[i][j];
        if(seq[i][j]<min)
          min = seq[i][j];
        if(seq[i][j]>0)
          overzero ++;
      }
      feature.push(max);
      feature.push(min);
      feature.push(overzero);
      feature.push(max-min);
    }
    return feature;
  },
  /*根据特征值计算四个动作的概率*/
  naive_calc_prob:function(fea){
    var mean = this.data.mean;
    var stdev = this.data.stdev;
    var rows = this.data.rows;
    var sum_row = 0;
    var prob = new Array(4);
    for(let i=0;i<4;i++){
      sum_row += rows[i];
    }
    for(let i=0;i<4;i++){
      prob[i] = rows[i]/sum_row;
      for(let j=0;j<fea.length;j++){
        if(stdev[i][j]!=0){
          prob[i] *= Math.exp(-((fea[j] - mean[i][j])*(fea[j] - mean[i][j]) / (
            2 * stdev[i][j] * stdev[i][j])));
          prob[i] *= (1/(Math.sqrt(2*Math.PI) * stdev[i][j]));
        }
        else if(fea[j]==mean[i][j])
            prob[i] *=1;
        else
            prob[i] *=0;
      }
    }
    return prob;
  },
  /*
    控制计数的整体逻辑
  */
  count: function(){
    var seq = new Array(6);//完整序列
    seq[0] = this.data.acc_x;
    seq[1] = this.data.acc_y;
    seq[2] = this.data.acc_z;
    seq[3] = this.data.gyr_x;
    seq[4] = this.data.gyr_y;
    seq[5] = this.data.gyr_z;
    var win = 32;//窗口大小
    var sub_seq = new Array(6);//长度为win的序列片段
    var old_motion = -1;
    var cut = new Array();//保存分段以后的序列
    var tag = new Array();//保存分段以后的各序列对应的动作
    var tank = new Array();//临时存储单个动作序列
    var cur_motion;

    /*将长序列分段*/
    //1.当此片段与上一片段动作相同时，将次片段连接到tank
    //2.当次片段对应的四个动作的概率都太小时，跳过
    //3.当次片段与上一个片段动作不同时，将tank推入cut
    //舍弃头部的win/2个数据
    for(let s = 0;(s+win)<seq[0].length;s+=parseInt(win/2)){
      for(let i=0;i<6;i++){
        sub_seq[i] = seq[i].slice(s,s+win);
      }
      var features = this.extract_features(sub_seq);
      var prob = this.naive_calc_prob(features);
      cur_motion = this.find_max(prob);
      if(s==0)
        old_motion = cur_motion;
      if(cur_motion == old_motion){
        tank = tank.concat(seq[this.data.axis_map[cur_motion]].slice(s+win/2,s+win));
      }
      else if(prob[cur_motion]<1e-100){
        continue;
      }
      else{
        cut.push(tank);
        tag.push(old_motion);
        old_motion = cur_motion;
        tank = [];
      }
    }
    //收尾
    cut.push(tank);
    tag.push(old_motion);

    /*逐个序列计算次数，并累加到tower*/
    var temp_count = 0;
    var tower = new Array(4).fill(0);//保存四个动作各自的次数

    for(let i=0;i<cut.length;i++){
      //序列长度太小直接舍弃
      if(cut[i].length <= 16){
        continue;
      }
      temp_count = this.calc_repetition(cut[i]);

      //针对缓冲深蹲跳波形特点---一个动作三个峰，作特别处理
      if(tag[i]==1)
        temp_count = parseInt(temp_count/3);
      tower[tag[i]] += temp_count;
      console.log("tower[",tag[i],"]: ",tower[tag[i]]);
    }

    return tower;
  },

  /*根据滤波后的极值点数计算动作次数*/
  calc_repetition: function(array){
    array = this.mid_smooth(array,9);
    array = this.mid_smooth(array,7);
    array = this.mid_smooth(array,5);
    var count_max = 0;
    var count_min = 0;
    //因动作开始和结束波形不稳，所以两边各剔除16个点
    for(let i=16;i<array.length-2-16;i++){
      if(array[i+1]>array[i] && array[i+1]>array[i+2])
        count_max++;
      if(array[i+1]<array[i] && array[i+1]<array[i+2])
        count_min++;
    }
    console.log("min & max: ",count_max,count_min);
    return  Math.min(count_max,count_min);
  },

  //寻找序列最大值对应的下标
  find_max: function(array){
    var max = -100;
    var index = -1;
    for(let i=0;i<array.length;i++){
      if(array[i]>max){
        index = i;
        max = array[i];
      }
    }
    return index;
  },
  
  //对序列施加宽度为span的中值滤波
  mid_smooth:function(arr,span){
    var mid = parseInt(span/2);
    var copy = arr.concat();
    for(let i=0;i<arr.length-span;i++){
      for(let j=0;j<span;j++){
        arr[i+mid] += copy[i+j];
      }
      arr[i+mid] /=  span;
    }
    
    for(let i=1;i<mid;i++){
      for(let j=-i;j<i;j++){
        arr[i] += copy[i+j];
        arr[-i] += copy[i+j]
      }
      arr[i] /= i;
      arr[-i] /= i;
    }

    return arr;
  },

  //为用户展示相应动作的演示图片
  demo:function(e){
    var id = e.currentTarget.dataset.id;
    var src = this.data.imgsrc[id];
    wx.navigateTo({
      url: '../demo/demo?src='+src,
    })
  }
})