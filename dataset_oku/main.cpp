#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "cifar10_reader.hpp"
#define MNIST_DATA_LOCATION "./fashion-mnist"
#include <stdlib.h>
#include <math.h>
#include "Neuron.h"
#include "Weights.h"
#include "Conv2D.h"
#include "Conv3D.h"
using namespace std;

float** iki_boyutlu(int rows, int columns)
{
    float** dizi;
    dizi = new float* [rows] {};
    for (int i = 0; i < rows; i++)
    {
        dizi[i] = new float[columns] {};
    }
    return dizi;
}
float** fashion_mnist_oku(int goruntu)
{
    float** input = iki_boyutlu(28, 28);
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    for (int row = 0; row < 28; row++)
    {
        for (int column = 0; column < 28; column++)
        {
            input[row][column] = unsigned(dataset.test_images[goruntu][(row * 28 + column)]);
        }
    }
    return input;
}

float*** uc_boyutlu(int rows, int columns, int depth)
{
    float*** dizi;
    dizi = new float** [rows] {};
    for (int i = 0; i < rows; i++)
    {
        dizi[i] = new float* [columns] {};
        for (int j = 0; j < columns; j++)
        {
            dizi[i][j] = new float[depth] {};
        }
    }
    return dizi;
}

float*** cifar_10_oku(int goruntu)
{
    float*** input = uc_boyutlu(32, 32, 3);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    int r, g, b;
    for (int row = 0; row < 32; row++)
    {
        for (int column = 0; column < 32; column++)
        {
            r = unsigned(dataset.training_images[goruntu][(row * 32 + column)]);
            g = unsigned(dataset.training_images[goruntu][(row * 32 + column) + 1024]);
            b = unsigned(dataset.training_images[goruntu][(row * 32 + column) + 2048]);
            input[row][column][0] = r;
            input[row][column][1] = g;
            input[row][column][2] = b;
        }
    }
    return input;
}
void maxPooling(float ***input, float ***output,int neuronSize, int height, int width)
{
    //cout<<"noron: "<<neuronSize<<"height:"<<height<<"width:"<<width<<endl;
    int bbb = 0, ccc = 0;
    for(int a=0; a<neuronSize; a++)   /// input derinlik
    {
        bbb=0;
        for(int b=0; b<height; b=b+2)  /// input height
        {
            ccc=0;
            for(int c=0; c<width; c=c+2)   /// input width
            {
                float buyuk = input[b][c][a];
                for(int bb=0; bb<2; bb++)
                {
                    for(int cc=0; cc<2; cc++)
                    {
                        if(input[b+bb][c+cc][a] > buyuk)
                        {
                            buyuk = input[b+bb][c+cc][a];
                        }
                    }
                }
                output[bbb][ccc][a] = buyuk;
                ccc++;
            }
            bbb++;
        }
    }

}
float*** arrayAllocate(int a, int b, int c)
{
    float ***input;
    input = new float**[a];

    for(int i=0; i<a; i++)
    {
        input[i]= new float*[b];

        for(int j=0; j<b; j++)
        {
            input[i][j]=new float[c];
        }
    }
    return input;
}
float**** arrayAllocate3d(int a, int b, int c, int d)
{
    float ****input;
    input = new float***[a];

    for(int i=0; i<a; i++)
    {
        input[i]= new float**[b];

        for(int j=0; j<b; j++)
        {
            input[i][j] = new float*[c];

            for(int k=0;k<c; k++){
                input[i][j][k]=new float[d];
            }
        }
    }
    return input;
}
float* flatten(float*** input, int width, int height, int depth)
{
    int arraySize = width * height * depth;
    float* flattenArray = new float[arraySize];
    int index = 0;
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            for(int k=0; k<depth; k++)
            {
                flattenArray[index++] = input[i][j][k];
            }
        }
    }
    return flattenArray;
}
float* flatten3d(float**** input, int width, int height, int depth, int frame)
{
    int arraySize = width * height * depth * frame;
    float* flattenArray = new float[arraySize];
    int index = 0;
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            for(int k=0; k<depth; k++)
            {
                for(int t=0; t<frame; t++)
                {
                    flattenArray[index++] = input[i][j][k][t];
                }
            }
        }
    }
    return flattenArray;
}
int main()
{
/*
    Conv3D conv3d;
    conv3d.setLayerArrayAllocate(5, 5, 3, 3, 8);
    conv3d.setLayerWeights("conv3d_kernel_1.txt", 5, 5, 3, 3, 8);
    conv3d.setBias("conv3d_bias_1.txt", 8);

    cout<<"conv"<<endl;
    fstream oku("c3d_input.txt");
    string row;
    float input[32][32][10][3];
    if(oku.is_open())
    {
        while(!oku.eof())
        {
            for(int a=0; a<32; a++)
            {
                for(int b=0; b<32; b++)
                {
                    for(int c=0; c<10; c++)
                    {
                        for(int d=0; d<3; d++)
                        {
                            oku >> row;
                            //cout<<row<<endl;
                            input[a][b][c][d] = atof(row.c_str());
                            //cout<<temp[a][b][c][d]<<endl;
                        }
                    }
                }
            }
            break;
        }
    }

    int bb = 0;
    int cc = 0;
    int dd = 0;
    float result_conv[28][28][8][8];
    for(int a=0; a<8; a++)  // nöron sayısı
    {
        bb = 0;
        for(int b=1; b<10-(1); b++)///input frame sayısı ikinci convda = 8 olacak
        {
            cc = 0;
            for(int c=2; c<32-(2); c++) ///input height
            {
                dd = 0;
                for(int d=2; d<32-(2); d++)/// input width
                {
                    float top=0;
                    for(int e=0; e<5; e++) /// mask height
                    {
                        for(int f=0; f<5; f++) /// mask width
                        {
                            for(int g=0; g<3; g++) /// mask  frame depth
                            {
                                for(int h=0; h<3; h++) /// mask depth 2. conv = 8 olacak
                                {
                                    top = top + input[c-2+e][d-2+f][b-1+g][h] * conv3d.getLayerWeights(e,f,g,h,a);
                                }
                            }
                        }
                    }
                    top = top+ conv3d.getBias(a);
                    //result_conv[cc][dd][bb][a] =top;
                    result_conv[c-2][d-2][b-1][a] =top;

                    dd = dd + 1;
                }
                cc = cc + 1;
            }
            bb = bb + 1;
        }
    }


    int bb3 = 0;
    int cc3 = 0;
    int dd3 = 0;
    float max_result[14][14][8][8];
    for(int a=0; a<8; a++)  /// nöron derinliği
    {
        bb3 = 0;
        for(int b=0; b<8; b++) /// input frame sayısı
        {
            cc3 = 0;
            for(int c=0; c<28-1; c=c+2)       /// input frame height
            {
                dd3 = 0;
                for(int d=0; d<28-1; d=d+2)     /// input frame width
                {
                    float enbuyuk=result_conv[c][d][b][a];
                    for(int bb=0; bb<1; bb++)    /// mask frame depth
                    {
                        for(int cc=0; cc<2; cc++)  /// mask height
                        {
                            for(int dd=0; dd<2; dd++)  /// mask weight
                            {
                                if(result_conv[c+cc][d+dd][b+bb][a] > enbuyuk)
                                {
                                    enbuyuk = result_conv[c+cc][d+dd][b+bb][a];
                                }
                            }
                        }
                    }
                    max_result[cc3][dd3][bb3][a] = enbuyuk;
                    dd3 = dd3 + 1;
                }
                cc3 = cc3 + 1;
            }
            bb3 = bb3 + 1;
        }
    }


    Conv3D conv2_3d;
    conv2_3d.setLayerArrayAllocate(5,5,3,8,16);
    conv2_3d.setLayerWeights("conv3d_kernel_2.txt", 5, 5, 3, 8, 16);
    conv2_3d.setBias("conv3d_bias_2.txt", 16);

    bb = 0;
    cc = 0;
    dd = 0;
    float result_conv2[10][10][6][16];
    for(int a=0; a<16; a++)  /// nöron sayısı
    {
        bb = 0;
        for(int b=1; b<8-(1); b++)///input frame sayısı ikinci convda = 8 olacak
        {
            cc = 0;
            for(int c=2; c<14-(2); c++) ///input height
            {
                dd = 0;
                for(int d=2; d<14-(2); d++)/// input width
                {
                    float top=0;
                    for(int e=0; e<5; e++) /// mask height
                    {
                        for(int f=0; f<5; f++) /// mask width
                        {
                            for(int g=0; g<3; g++) /// mask  frame depth
                            {
                                for(int h=0; h<8; h++) /// mask depth 2. conv = 8 olacak
                                {
                                    top = top + max_result[c-2+e][d-2+f][b-1+g][h] * conv2_3d.getLayerWeights(e,f,g,h,a);
                                }
                            }
                        }
                    }
                    top = top + conv2_3d.getBias(a);
                    result_conv2[c-2][d-2][b-1][a] = top;
                    dd = dd + 1;
                }
                cc = cc + 1;
            }
            bb = bb + 1;
        }
    }


    bb3 = 0;
    cc3 = 0;
    dd3 = 0;
    float ****max_result_2 = arrayAllocate3d(5, 5, 3, 16);

    ///float max_result_2[5][5][3][16];
    for(int a=0; a<16; a++)  /// nöron derinliği
    {
        bb3 = 0;
        for(int b=0; b<3; b++) /// input frame sayısı
        {
            cc3 = 0;
            for(int c=0; c<10; c=c+2)       /// input frame height
            {
                dd3 = 0;
                for(int d=0; d<10; d=d+2)     /// input frame width
                {
                    float enbuyuk=result_conv2[c][d][b][a];
                    for(int bb=0; bb<2; bb++)    /// mask frame depth
                    {
                        for(int cc=0; cc<2; cc++)  /// mask height
                        {
                            for(int dd=0; dd<2; dd++)  /// mask weight
                            {
                                if(result_conv2[c+cc][d+dd][b+bb][a] > enbuyuk)
                                {
                                    enbuyuk = result_conv2[c+cc][d+dd][b+bb][a];
                                }
                            }
                        }
                    }
                    //cout<<"cc3 : ["<<cc3<<"]  dd3: ["<<dd3<<"]   bb3: ["<<bb3<<"]  a : "<<a<<endl;
                    max_result_2[cc3][dd3][bb3][a] = enbuyuk;

                    dd3 = dd3 + 1;
                }
                cc3 = cc3 + 1;
            }
            bb3 = bb3 + 1;
        }
    }
ofstream bakk("yaz.txt");
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            for(int k=0; k<3; k++)
            {
                for(int m=0; m<16; m++)
                {
                    bakk<<max_result_2[i][j][k][m]<<"  ";
                }
                bakk<<endl;
            }
            bakk<<endl;
        }
    }
    Weights convWeights(32,38400,16,512);
    convWeights.setFirstLayerBias("conv3d_fc_bias_1.txt");
    convWeights.setFirstLayerWeight("conv3d_fc_kernel_1.txt");
    convWeights.setSecondLayerBias("conv3d_fc_bias_2.txt");
    convWeights.setSecondLayerWeight("conv3d_fc_kernel_2.txt");

    float firstTemp[32] = { 0 } , secondTemp[16] = { 0 };

        Neuron firstLayerNeuron;
        int firstLayerNeuronSize = 32;
        int counter=0, index;
        float *flattenArray = flatten3d(max_result_2,5,5,3,16);

        ofstream yazz("yaz.txt");
        for(int id = 0; id < firstLayerNeuronSize; id++) /// first layer neuron size
        {
            counter = id;
            index = 0;
            firstLayerNeuron.setId(id);
            for(int i = 0; i < 1200; i++)
            {
                firstTemp[id] = firstTemp[id] + (flattenArray[i] *
                                                     convWeights.getFirstLayerWeight((counter + index*firstLayerNeuronSize)));
                //yaz<< flattenArray[i]<<"   "<<convWeights.getFirstLayerWeight((counter + index*32))<<endl;
                index++;
            }
            firstTemp[id] +=  convWeights.getFirstLayerBias(id);
            firstLayerNeuron.setValue(firstTemp[id], id);

        }

        Neuron secondLayerNeuron;
        int secondLayerNeuronSize = 16;
        for(int id = 0; id < secondLayerNeuronSize; id++){ /// second layer neuron size

            counter = id;
            index = 0;
            for(int j = 0; j < firstLayerNeuronSize; j++)
            {
               secondTemp[id] = secondTemp[id] + (firstLayerNeuron.getValue(j) *
                                                  convWeights.getSecondLayerWeight((counter + index*secondLayerNeuronSize)));
               index++;
            }
            secondTemp[id] += convWeights.getSecondLayerBias(id);
            secondLayerNeuron.setId(id);
            secondLayerNeuron.setValue(secondTemp[id], id);
        }
    Weights w1(1,512,44,44);
    w1.setFirstLayerBias("conv3d_fc_bias_3.txt");
    w1.setFirstLayerWeight("conv3d_fc_kernel_3.txt");
    float resultFc = 0;
    for(int i=0;i<16;i++)
    {
        resultFc += secondLayerNeuron.getValue(i) +
                            w1.getFirstLayerWeight(i);
    }
    resultFc += w1.getFirstLayerBias(0);
    cout<<"result: "<<resultFc<<endl;
*/

    Conv2D conv;

    conv.setLayerArrayAllocate(3, 3, 3, 32);

    conv.setLayerWeights("conv_kernel1.txt", 3, 3, 3, 32);

    conv.setBias("conv_bias1.txt", 32);

    float*** resim = cifar_10_oku(7);
    //float result[30][30][32];
    float ***result = arrayAllocate(30,30,32);

    ofstream yaz_carpim("result_carpim.txt");
    for(int m=0; m<32; m++)    /// noron sayısı
    {
        for(int a=1; a<32-1; a++)  /// katmana gelen veri Height
        {
            for(int b=1; b<32-1; b++)  /// katmana gelen veri Width
            {
                float sum =0;
                for(int d=0; d<3; d++)   /// noron height
                {
                    for(int e=0; e<3; e++) /// noron width
                    {
                        for(int f=0; f<3; f++)  /// noron derinlik
                        {
                            sum += resim[a-1+d][b-1+e][f] * conv.getLayerWeights(d,e,f,m);
                        }
                    }
                }
                sum +=conv.getBias(m);
                result[a-1][b-1][m] = sum;
            }
        }
    }

    //float maxResult[15][15][32];    /// MaxPooling
    float ***maxResult = arrayAllocate(15, 15, 32);    /// MaxPooling

    maxPooling(result, maxResult, 32, 30, 30);

    Conv2D conv2;
    conv2.setLayerArrayAllocate(3, 3, 32, 64);
    conv2.setLayerWeights("conv_kernel2.txt", 3, 3, 32, 64);
    conv2.setBias("conv_bias2.txt", 64);

    //float result_2[13][13][64];
    float ***result_2 = arrayAllocate(13, 13, 64);

    for(int m=0; m<64; m++)    /// noron sayisi
    {
        for(int a=1; a<15-1; a++)  /// katmana gelen veri Height
        {
            for(int b=1; b<15-1; b++)  /// katmana gelen veri Width
            {
                float sum =0;
                for(int d=0; d<3; d++)   /// noron height
                {
                    for(int e=0; e<3; e++) /// noron width
                    {
                        for(int f=0; f<32; f++)  /// noron derinlik
                        {
                            sum += maxResult[a-1+d][b-1+e][f] * conv2.getLayerWeights(d,e,f,m);
                            //cout<<maxResult[a-1+d][b-1+e][f]<<endl;
                        }
                    }
                }
                sum += conv2.getBias(m);
                result_2[a-1][b-1][m] = sum;
            }
        }
    }

    float ***maxResult_2 = arrayAllocate(6, 6, 64);    /// MaxPooling

    maxPooling(result_2, maxResult_2, 64, 12, 12);

    //float result_3[4][4][64];
    float ***result_3 = arrayAllocate(4, 4, 64);

    Conv2D conv3;
    conv3.setLayerArrayAllocate(3, 3, 64, 64);
    conv3.setLayerWeights("conv_kernel3.txt", 3, 3, 64, 64);
    conv3.setBias("conv_bias3.txt", 64);

    for(int m=0; m<64; m++)    /// noron sayisi
    {
        for(int a=1; a<6-1; a++)  /// katmana gelen veri Height
        {
            for(int b=1; b<6-1; b++)  /// katmana gelen veri Width
            {
                float sum = 0;
                for(int d=0; d<3; d++)   /// noron height
                {
                    for(int e=0; e<3; e++) /// noron width
                    {
                        for(int f=0; f<64; f++)  /// noron derinlik
                        {
                            sum += maxResult_2[a-1+d][b-1+e][f] * conv3.getLayerWeights(d,e,f,m);
                        }
                    }
                }
                sum += conv3.getBias(m);
                result_3[a-1][b-1][m] = sum;
            }
        }
    }


    ofstream bakk("maxResult.txt");
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            for(int k=0; k<64; k++)
            {
                bakk<<result_3[i][j][k]<<"   ";
            }
            bakk<<endl;
        }
        bakk<<endl;
        bakk<<endl;
    }


        float* flattenArray = flatten(result_3, 4,4,64);


        Weights convWeights(64, 65536, 10, 640);
        convWeights.setFirstLayerBias("fully_bias_1.txt");
        convWeights.setFirstLayerWeight("fully_kernel_1.txt");
        convWeights.setSecondLayerBias("fully_bias_2.txt");
        convWeights.setSecondLayerWeight("fully_kernel_2.txt");

        float firstTemp[64] = { 0 } , secondTemp[10] = { 0 };


        Neuron firstLayerNeuron;
        int firstLayerNeuronSize = 64;

        int counter=0, index;
        ofstream yazz("yaz.txt");
        float sonuc[64];
        for(int id = 0; id < 64; id++) /// first layer neuron size
        {
            counter = id;
            index = 0;
            firstLayerNeuron.setId(id);
            for(int i = 0; i < 1024; i++)
            {
                firstTemp[id] = firstTemp[id] + (flattenArray[i] *
                                                     convWeights.getFirstLayerWeight((counter + index*64)));
                yazz<< flattenArray[i]<<"   "<<convWeights.getFirstLayerWeight((counter + index*64))<<endl;
                index++;
            }
            firstTemp[id] +=  convWeights.getFirstLayerBias(id);
            firstLayerNeuron.setValue(firstTemp[id], id);

        }
        ofstream yazzz("first_layer.txt");
        for(int i= 0; i<firstLayerNeuronSize; i++){
           yazzz<<firstLayerNeuron.getValue(i)<<endl;
        }
        Neuron secondLayerNeuron;
        int secondLayerNeuronSize = 10;
        for(int id = 0; id < secondLayerNeuronSize; id++){ /// second layer neuron size

            counter = id;
            index = 0;
            for(int j = 0; j < firstLayerNeuronSize; j++)
            {
                //cout<<sonuc[j]<<"   "<<firstLayerNeuron.getValue(j)<<endl;
               secondTemp[id] = secondTemp[id] + (firstLayerNeuron.getValue(j) *
                                                  convWeights.getSecondLayerWeight((counter + index*10)));
               index++;
            }
            secondTemp[id] += convWeights.getSecondLayerBias(id);
            secondLayerNeuron.setId(id);
            secondLayerNeuron.setValue(secondTemp[id], id);
            cout<<secondTemp[id]<<endl;
        }


         //for(int i = 0; i < secondLayerNeuronSize; i++)
            //cout<<secondLayerNeuron.getValue(i)<<endl;
        /*
        float** image = fashion_mnist_oku(0);

        Weights weights(128, 100352, 10, 1280);

        weights.setFirstLayerBias("bias_1.txt");
        weights.setSecondLayerBias("bias_2.txt");
        weights.setFirstLayerWeight("first_weights.txt");
        weights.setSecondLayerWeight("second_weights.txt");


        float tempArray[128] = {0},    tempArray2[10] = {0};
        Neuron firstLayerNeuron;

        int counter=0, index;
        for(int id = 0; id < 128; id++) /// first layer neuron size
        {
            counter = id;
            index = 0;
            firstLayerNeuron.setId(id);
            for(int i = 0; i < 28; i++)
            {
                for(int j = 0; j < 28; j++)
                {
                    tempArray[id] = tempArray[id] + (image[i][j] *
                                                     weights.getFirstLayerWeight((counter + index*128)));
                    index++;
                }
            }
            tempArray[id] +=  weights.getFirstLayerBias(id);
            firstLayerNeuron.setValue(tempArray[id], id);
        }

        Neuron secondLayerNeuron;

        for(int id = 0; id < 10; id++){ /// second layer neuron size

            counter = id;
            index = 0;
            for(int j = 0; j < 128; j++)
            {
               tempArray2[id] = tempArray2[id] + (firstLayerNeuron.getValue(j) *
                                                  weights.getSecondLayerWeight((counter + index*10)));
               index++;
            }
            tempArray2[id] += weights.getSecondLayerBias(id);
            secondLayerNeuron.setId(id);
            secondLayerNeuron.setValue(tempArray2[id], id);
        }
        for(int i = 0;i<10;i++)
            cout<<secondLayerNeuron.getValue(i)<<endl;
    */
}
