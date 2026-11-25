import {writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
const root=dirname(import.meta.path);
const ink='#243447', excitation='#b74343', inhibition='#2563a6';
const label=(x:number,y:number,value:string,color=ink)=>`<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" font-family="Arial, Helvetica, sans-serif" font-size="22" fill="${color}">${value}</text>`;
const cell=(x:number,y:number,value:string,color=ink)=>`<circle cx="${x}" cy="${y}" r="33" fill="white" stroke="${color}" stroke-width="2.5"/>${label(x,y,value,color)}`;
const path=(d:string,color=ink)=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;
function save(file:string,title:string,body:string){writeFileSync(join(root,file),`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="270" viewBox="0 0 640 270" role="img" aria-labelledby="title"><title id="title">${title}</title><rect width="640" height="270" fill="white"/>${body}</svg>\n`);}
save('gamma-circuit.svg','Excitatory and inhibitory populations with feedback and recurrent inhibition',
 cell(170,135,'E',excitation)+cell(390,135,'I',inhibition)+
 path('M203 119 H354 M344 112 L354 119 L344 126',excitation)+
 path('M356 151 H204 M204 142 V160',inhibition)+
 path('M411 109 C498 42 510 222 413 163 M409 155 L419 171',inhibition));
const wave=(x:number,y:number,phase:number,color:string)=>path(Array.from({length:81},(_,i)=>`${i?'L':'M'}${x+i*1.25},${y-15*Math.cos(i*Math.PI*4/80+phase)}`).join(' '),color);
save('phase-routing.svg','Two rhythmic input streams converge on a receiver',
 cell(260,75,'A',excitation)+cell(260,195,'B',inhibition)+cell(475,135,'R')+
 wave(55,75,0,excitation)+wave(55,195,Math.PI,inhibition)+
 path('M294 82 L438 125 M426 116 L438 125 L423 126',excitation)+
 path('M294 188 L438 145 M423 144 L438 145 L426 154',inhibition)+
 wave(510,135,0,ink));
console.log('Created two minimal standalone diagrams.');
